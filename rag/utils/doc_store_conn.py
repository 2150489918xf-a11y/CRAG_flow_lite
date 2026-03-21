"""
文档存储连接器抽象基类 (DocStoreConnection)
所有上层模块（search, graph, api）只依赖此抽象接口，
具体的存储后端（ES, Milvus, Qdrant, FAISS 等）作为可插拔实现注入。
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DocStoreConnection(ABC):
    """向量/文档存储引擎的统一抽象接口"""

    # ──────────────── 生命周期 ────────────────

    @abstractmethod
    def health(self) -> dict:
        """检查后端存储健康状态"""
        ...

    # ──────────────── 索引管理 ────────────────

    @abstractmethod
    def create_idx(self, index_name: str, mapping_path: str = None, display_name: str = None, folder: str = "/") -> bool:
        """创建索引/Collection"""
        ...

    @abstractmethod
    def delete_idx(self, index_name: str) -> bool:
        """删除索引/Collection"""
        ...

    @abstractmethod
    def index_exist(self, index_name: str) -> bool:
        """判断索引是否存在"""
        ...

    @abstractmethod
    def get_index_meta(self, index_name: str) -> dict:
        """获取索引的元信息（如 display_name, folder 等）"""
        ...

    @abstractmethod
    def update_index_meta(self, index_name: str, **meta_fields):
        """更新索引的元信息（增量合并）"""
        ...

    @abstractmethod
    def list_indices(self, prefix: str = "ragflow_lite_*") -> dict[str, dict]:
        """
        列出匹配前缀的所有索引。
        Returns: {index_name: {info...}, ...}
        """
        ...

    @abstractmethod
    def count_docs(self, index_name: str) -> int:
        """获取索引中的文档数量"""
        ...

    @abstractmethod
    def search_raw(self, index_name: str, body: dict) -> dict:
        """执行原始查询（用于文档列表等管理操作）"""
        ...

    @abstractmethod
    def delete_by_query(self, index_name: str, body: dict) -> int:
        """按原始查询条件删除文档，返回删除数量"""
        ...

    @abstractmethod
    def refresh_index(self, index_name: str):
        """刷新索引使最近的写入可被搜索到"""
        ...

    # ──────────────── 文档 CRUD ────────────────

    @abstractmethod
    def insert(self, documents: List[dict], index_name: str) -> List[str]:
        """
        批量插入文档。
        Returns: 错误信息列表，空列表表示全部成功。
        """
        ...

    @abstractmethod
    def delete(self, condition: dict, index_name: str) -> int:
        """
        按条件删除文档。
        Returns: 被删除的文档数量。
        """
        ...

    @abstractmethod
    def get_by_ids(self, ids: list, index_name: str, source_fields: list = None) -> Dict[str, dict]:
        """
        批量获取文档。
        Returns: {doc_id: {field: value, ...}, ...}
        """
        ...

    # ──────────────── 搜索 ────────────────

    @abstractmethod
    def search(self, select_fields, highlight_fields, condition, match_expressions,
               offset, limit, index_names, rank_feature=None, exclude_parent=False) -> Any:
        """
        混合检索（全文 + 向量）。
        返回值为后端原生结果，由 get_total/get_source 等方法统一解析。
        """
        ...

    # ──────────────── 结果解析器 ────────────────

    @staticmethod
    @abstractmethod
    def get_total(res) -> int:
        """从搜索结果中提取命中总数"""
        ...

    @staticmethod
    @abstractmethod
    def get_doc_ids(res) -> List[str]:
        """从搜索结果中提取文档 ID 列表"""
        ...

    @staticmethod
    @abstractmethod
    def get_source(res) -> List[dict]:
        """从搜索结果中提取文档源数据列表"""
        ...

    @staticmethod
    @abstractmethod
    def get_fields(res, fields) -> Dict[str, dict]:
        """从搜索结果中提取指定字段"""
        ...

    @staticmethod
    @abstractmethod
    def get_highlight(res, keywords=None, field="content_with_weight") -> Dict[str, str]:
        """从搜索结果中提取高亮片段"""
        ...


# ──────────────── 工厂函数 ────────────────

_instance: Optional[DocStoreConnection] = None


def get_doc_store(**kwargs) -> DocStoreConnection:
    """
    获取文档存储连接器单例。
    通过 service_conf.yaml 的 doc_store.backend 配置切换后端。
    新增后端只需:
      1. 继承 DocStoreConnection
      2. @doc_store_registry.register("your_backend")
      3. 在 service_conf.yaml 设置 doc_store.backend: your_backend
    """
    global _instance
    if _instance is not None:
        return _instance

    from common.registry import doc_store_registry
    from rag.settings import get_config

    # 确保内置后端已注册 (触发 es_conn 模块加载)
    import rag.utils.es_conn  # noqa: F401

    cfg = get_config()
    backend = cfg.get("doc_store", {}).get("backend", "elasticsearch")

    _instance = doc_store_registry.create(backend, **kwargs)
    logger.info(f"DocStore backend initialized: {backend}")
    return _instance
