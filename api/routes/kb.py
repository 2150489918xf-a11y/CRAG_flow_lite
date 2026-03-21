"""
知识库管理路由 (Knowledge Base CRUD + 文件夹管理 + Stats)
"""
import json
import logging
import os
import re
import uuid

from fastapi import APIRouter

from api.deps import get_es, get_config
from api.models import (
    KnowledgeBaseCreate, BatchDeleteRequest,
    KBMoveRequest, FolderCreateRequest,
)
from api.errors import NotFoundError, ValidationError, ExternalServiceError, ok_response
from rag.nlp.search import index_name
from common.perf import perf

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["知识库管理"])

# ── 文件夹元数据存储路径 ──
_FOLDERS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "conf", "folders.json",
)


def _normalize_folder(path: str) -> str:
    """标准化文件夹路径: 确保以 / 开头，不以 / 结尾（根目录除外）"""
    path = path.strip().replace("\\", "/")
    if not path.startswith("/"):
        path = "/" + path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return path


def _load_folders() -> set[str]:
    """加载已创建的文件夹集合"""
    if os.path.isfile(_FOLDERS_FILE):
        try:
            with open(_FOLDERS_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return {"/"}


def _save_folders(folders: set[str]):
    """保存文件夹集合"""
    folders.add("/")  # 根目录始终存在
    os.makedirs(os.path.dirname(_FOLDERS_FILE), exist_ok=True)
    with open(_FOLDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(folders), f, ensure_ascii=False, indent=2)


def _ensure_folder_ancestors(path: str, folders: set[str]):
    """确保路径的所有祖先文件夹都存在"""
    parts = path.strip("/").split("/")
    current = ""
    for part in parts:
        current += "/" + part
        folders.add(current)


# ══════════════════════════════════════════
#  健康检查
# ══════════════════════════════════════════

@router.get("/health")
async def health():
    try:
        info = get_es().health()
        cfg = get_config()
        return ok_response({
            "es_status": info.get("status", "unknown"),
            "graph_enabled": cfg.get("graph", {}).get("enabled", False),
            "crag_enabled": cfg.get("crag", {}).get("enabled", False),
            "reranker_enabled": cfg.get("reranker", {}).get("enabled", False),
        })
    except Exception as e:
        raise ExternalServiceError("Elasticsearch", str(e))


# ══════════════════════════════════════════
#  知识库 CRUD
# ══════════════════════════════════════════

@router.post("/knowledgebase")
async def create_knowledgebase(req: KnowledgeBaseCreate):
    """创建知识库（ES 索引），支持中文名和文件夹归属"""
    display_name = req.kb_id.strip()
    if not display_name:
        raise ValidationError("知识库名称不能为空")

    folder = _normalize_folder(req.folder)

    safe_id = re.sub(r'[^a-z0-9_]', '', req.kb_id.lower().replace(' ', '_').replace('-', '_'))
    if not safe_id:
        safe_id = "kb_" + uuid.uuid4().hex[:8]

    es = get_es()
    idx = index_name(safe_id)

    # Check if an index with this display name already exists
    try:
        existing = es.es.indices.get(index="ragflow_lite_*")
        for existing_idx in existing:
            meta = es.get_index_meta(existing_idx)
            if meta.get("display_name") == display_name:
                existing_kb_id = existing_idx.replace("ragflow_lite_", "")
                return ok_response({
                    "kb_id": existing_kb_id, "display_name": display_name,
                    "folder": meta.get("folder", "/"),
                    "index": existing_idx,
                }, message="exists")
    except Exception:
        pass

    if es.index_exist(idx):
        return ok_response({"kb_id": safe_id, "display_name": display_name,
                            "folder": folder, "index": idx},
                           message="exists")

    # 确保目标文件夹存在
    folders = _load_folders()
    _ensure_folder_ancestors(folder, folders)
    _save_folders(folders)

    es.create_idx(idx, display_name=display_name, folder=folder)
    return ok_response({"kb_id": safe_id, "display_name": display_name,
                        "folder": folder, "index": idx},
                       message="created")


@router.get("/knowledgebase")
async def list_knowledgebases(folder: str = None):
    """
    列出知识库。
    - 不传 folder: 返回全部
    - folder="/": 只返回根目录下的 KB
    - folder="/财务": 返回该文件夹及子文件夹下的全部 KB
    """
    es = get_es()
    try:
        indices = es.es.indices.get(index="ragflow_lite_*")
        kbs = []
        for idx_name_str, info in indices.items():
            kb_id = idx_name_str.replace("ragflow_lite_", "")
            count = es.es.count(index=idx_name_str)["count"]
            meta = es.get_index_meta(idx_name_str)
            display_name = meta.get("display_name", kb_id)
            kb_folder = meta.get("folder", "/")

            # 文件夹过滤
            if folder is not None:
                norm = _normalize_folder(folder)
                if norm == "/":
                    pass  # 根目录显示全部
                elif kb_folder != norm and not kb_folder.startswith(norm + "/"):
                    continue

            kbs.append({
                "kb_id": kb_id,
                "display_name": display_name,
                "folder": kb_folder,
                "index": idx_name_str,
                "doc_count": count,
            })
        return ok_response({"knowledgebases": kbs})
    except Exception as e:
        raise ExternalServiceError("Elasticsearch", str(e))


@router.delete("/knowledgebase/{kb_id}")
async def delete_knowledgebase(kb_id: str):
    """删除知识库"""
    es = get_es()
    idx = index_name(kb_id)
    if es.delete_idx(idx):
        return ok_response({"kb_id": kb_id}, message="deleted")
    raise NotFoundError("知识库", kb_id)


@router.post("/knowledgebase/batch_delete")
async def batch_delete_knowledgebases(req: BatchDeleteRequest):
    """批量删除知识库"""
    if not req.kb_ids:
        raise ValidationError("kb_ids 不能为空")

    es = get_es()
    results = []
    deleted_count = 0
    failed_count = 0

    for kb_id in req.kb_ids:
        idx = index_name(kb_id)
        try:
            if es.delete_idx(idx):
                results.append({"kb_id": kb_id, "status": "deleted"})
                deleted_count += 1
            else:
                results.append({"kb_id": kb_id, "status": "not_found"})
                failed_count += 1
        except Exception as e:
            results.append({"kb_id": kb_id, "status": "error", "detail": str(e)})
            failed_count += 1
            logger.error(f"Failed to delete kb '{kb_id}': {e}")

    return ok_response({
        "deleted": deleted_count,
        "failed": failed_count,
        "results": results,
    })


# ══════════════════════════════════════════
#  文件夹管理
# ══════════════════════════════════════════

@router.post("/knowledgebase/folder")
async def create_folder(req: FolderCreateRequest):
    """创建文件夹（支持嵌套，自动创建父级）"""
    path = _normalize_folder(req.path)
    if path == "/":
        raise ValidationError("根目录已存在")

    folders = _load_folders()
    _ensure_folder_ancestors(path, folders)
    _save_folders(folders)

    return ok_response({"folder": path}, message="created")


@router.delete("/knowledgebase/folder")
async def delete_folder(path: str):
    """
    删除文件夹（仅当文件夹下没有知识库时）
    """
    path = _normalize_folder(path)
    if path == "/":
        raise ValidationError("不能删除根目录")

    # 检查是否有 KB 在该文件夹下
    es = get_es()
    try:
        indices = es.es.indices.get(index="ragflow_lite_*")
        for idx_name_str in indices:
            meta = es.get_index_meta(idx_name_str)
            kb_folder = meta.get("folder", "/")
            if kb_folder == path or kb_folder.startswith(path + "/"):
                raise ValidationError(f"文件夹 '{path}' 下还有知识库，无法删除")
    except ValidationError:
        raise
    except Exception:
        pass

    folders = _load_folders()
    # 删除该文件夹及子文件夹
    to_remove = {f for f in folders if f == path or f.startswith(path + "/")}
    folders -= to_remove
    _save_folders(folders)

    return ok_response({"folder": path, "removed": len(to_remove)}, message="deleted")


@router.post("/knowledgebase/move")
async def move_knowledgebase(req: KBMoveRequest):
    """移动知识库到目标文件夹"""
    target = _normalize_folder(req.target_folder)

    es = get_es()
    idx = index_name(req.kb_id)
    if not es.index_exist(idx):
        raise NotFoundError("知识库", req.kb_id)

    # 确保目标文件夹存在
    folders = _load_folders()
    _ensure_folder_ancestors(target, folders)
    _save_folders(folders)

    es.update_index_meta(idx, folder=target)

    return ok_response({
        "kb_id": req.kb_id,
        "folder": target,
    }, message="moved")


@router.get("/knowledgebase/tree")
async def get_folder_tree():
    """
    返回文件夹树形结构 + 各文件夹下的知识库

    返回格式:
    {
      "tree": [
        {"path": "/", "name": "/", "children": [...], "kbs": [...]},
        ...
      ]
    }
    """
    folders = _load_folders()

    # 获取所有 KB 及其文件夹归属
    es = get_es()
    kb_map = {}  # folder -> [kb_info, ...]
    try:
        indices = es.es.indices.get(index="ragflow_lite_*")
        for idx_name_str in indices:
            kb_id = idx_name_str.replace("ragflow_lite_", "")
            meta = es.get_index_meta(idx_name_str)
            kb_folder = meta.get("folder", "/")
            count = es.es.count(index=idx_name_str)["count"]

            # 确保 KB 的文件夹在 folders 中
            folders.add(kb_folder)

            kb_info = {
                "kb_id": kb_id,
                "display_name": meta.get("display_name", kb_id),
                "doc_count": count,
            }
            kb_map.setdefault(kb_folder, []).append(kb_info)
    except Exception:
        pass

    # 构建树
    def build_tree(path: str) -> dict:
        name = path.rsplit("/", 1)[-1] if path != "/" else "/"
        children = []
        for f in sorted(folders):
            if f == path:
                continue
            # f 是 path 的直接子文件夹
            parent = f.rsplit("/", 1)[0] if "/" in f[1:] else "/"
            if parent == path:
                children.append(build_tree(f))
        return {
            "path": path,
            "name": name,
            "children": children,
            "kbs": kb_map.get(path, []),
        }

    tree = build_tree("/")

    return ok_response({"tree": tree})


# ══════════════════════════════════════════
#  性能统计
# ══════════════════════════════════════════

@router.get("/stats")
async def get_stats():
    """获取管道性能统计（各阶段耗时 P50/P95/Avg/Max）"""
    return ok_response(perf.get_stats())


@router.post("/stats/reset")
async def reset_stats():
    """重置性能统计"""
    perf.reset()
    return ok_response(message="stats reset")
