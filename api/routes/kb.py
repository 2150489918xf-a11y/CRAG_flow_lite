"""
知识库管理路由 (Knowledge Base CRUD)
"""
import logging
import re
import uuid

from fastapi import APIRouter

from api.deps import get_es, get_config
from api.models import KnowledgeBaseCreate, BatchDeleteRequest
from api.errors import NotFoundError, ValidationError, ExternalServiceError, ok_response
from rag.nlp.search import index_name

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["知识库管理"])


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


@router.post("/knowledgebase")
async def create_knowledgebase(req: KnowledgeBaseCreate):
    """创建知识库（ES 索引），支持中文名"""
    display_name = req.kb_id.strip()
    if not display_name:
        raise ValidationError("知识库名称不能为空")

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
                    "index": existing_idx,
                }, message="exists")
    except Exception:
        pass

    if es.index_exist(idx):
        return ok_response({"kb_id": safe_id, "display_name": display_name, "index": idx},
                           message="exists")

    es.create_idx(idx, display_name=display_name)
    return ok_response({"kb_id": safe_id, "display_name": display_name, "index": idx},
                       message="created")


@router.get("/knowledgebase")
async def list_knowledgebases():
    """列出所有知识库"""
    es = get_es()
    try:
        indices = es.es.indices.get(index="ragflow_lite_*")
        kbs = []
        for idx_name_str, info in indices.items():
            kb_id = idx_name_str.replace("ragflow_lite_", "")
            count = es.es.count(index=idx_name_str)["count"]
            meta = es.get_index_meta(idx_name_str)
            display_name = meta.get("display_name", kb_id)
            kbs.append({
                "kb_id": kb_id,
                "display_name": display_name,
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
