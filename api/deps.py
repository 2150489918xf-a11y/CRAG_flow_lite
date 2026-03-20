"""
共享依赖注入层
所有 Router 通过此模块获取全局单例（DocStore, Embedding, Dealer 等）
"""
import logging
import os

from rag.utils.doc_store_conn import get_doc_store
from rag.llm.base import get_chat_client, get_embedding, get_reranker as get_reranker_factory
from rag.nlp.search import Dealer, index_name
from rag.settings import get_embedding_config, get_config

logger = logging.getLogger(__name__)

# 全局单例
_doc_store = None
_emb_mdl = None
_dealer = None
_graph_searcher = None
_reranker = None
_crag_router = None
_query_enhancer = None


def get_es():
    global _doc_store
    if _doc_store is None:
        _doc_store = get_doc_store()
    return _doc_store


def get_emb():
    global _emb_mdl
    if _emb_mdl is None:
        _emb_mdl = get_embedding()
    return _emb_mdl


def get_dealer():
    global _dealer
    if _dealer is None:
        _dealer = Dealer(get_es())
    return _dealer


def get_graph_searcher():
    global _graph_searcher
    if _graph_searcher is None:
        cfg = get_config()
        graph_cfg = cfg.get("graph", {})
        if not graph_cfg.get("enabled", False):
            return None
        from rag.graph.graph_search import GraphSearcher
        from rag.graph.graph_store import GraphStore
        chat = get_chat_client()
        graph_store = GraphStore(es_conn=get_es(), emb_mdl=get_emb())

        # 尝试加载已有的图文件
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "graphs")
        if os.path.isdir(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith("_graph.json"):
                    graph_store.load_graph(os.path.join(data_dir, f))

        _graph_searcher = GraphSearcher(
            es_conn=get_es(), emb_mdl=get_emb(),
            chat_client=chat, graph_store=graph_store,
        )
    return _graph_searcher


def get_reranker():
    global _reranker
    if _reranker is None:
        cfg = get_config()
        reranker_cfg = cfg.get("reranker", {})
        if not reranker_cfg.get("enabled", False):
            return None
        _reranker = get_reranker_factory()
    return _reranker


def get_crag_router():
    global _crag_router
    if _crag_router is None:
        cfg = get_config()
        if not cfg.get("crag", {}).get("enabled", False):
            return None
        from rag.crag.router import CRAGRouter
        _crag_router = CRAGRouter(chat_client=get_chat_client())
    return _crag_router


def get_query_enhancer():
    global _query_enhancer
    if _query_enhancer is None:
        from rag.nlp.query_enhance import QueryEnhancer
        _query_enhancer = QueryEnhancer(chat_client=get_chat_client())
    return _query_enhancer
