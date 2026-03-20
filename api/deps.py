"""
共享依赖注入层
所有 Router 通过此模块获取全局单例（DocStore, Embedding, Dealer 等）
"""
import logging
import os

from rag.utils.doc_store_conn import get_doc_store
from rag.llm.embedding import RemoteEmbedding
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
        cfg = get_embedding_config()
        _emb_mdl = RemoteEmbedding(
            api_key=cfg["api_key"],
            model_name=cfg["model_name"],
            base_url=cfg.get("base_url", ""),
        )
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
        from rag.llm.chat import ChatClient
        from rag.graph.graph_search import GraphSearcher
        from rag.graph.graph_store import GraphStore
        chat = ChatClient()
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
        from rag.llm.reranker import BGEReranker
        _reranker = BGEReranker(
            model_name=reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"),
        )
    return _reranker


def get_crag_router():
    global _crag_router
    if _crag_router is None:
        cfg = get_config()
        if not cfg.get("crag", {}).get("enabled", False):
            return None
        from rag.crag.router import CRAGRouter
        from rag.llm.chat import ChatClient
        _crag_router = CRAGRouter(chat_client=ChatClient())
    return _crag_router


def get_query_enhancer():
    global _query_enhancer
    if _query_enhancer is None:
        from rag.nlp.query_enhance import QueryEnhancer
        from rag.llm.chat import ChatClient
        _query_enhancer = QueryEnhancer(chat_client=ChatClient())
    return _query_enhancer
