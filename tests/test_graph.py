"""
GraphRAG 模块测试：数据结构、NetworkX 图、PageRank、格式化输出
（纯离线测试，不需要 ES 或 LLM）
"""
import os
import tempfile
import pytest
import networkx as nx


class TestGraphDataStructures:
    """图谱数据结构测试"""

    def test_entity_creation(self):
        from rag.graph.extractor import Entity
        e = Entity(name="微软", type="ORGANIZATION", description="科技公司")
        assert e.name == "微软"
        assert e.type == "ORGANIZATION"

    def test_relation_creation(self):
        from rag.graph.extractor import Relation
        r = Relation(source="微软", target="OpenAI", description="投资")
        assert r.source == "微软"
        assert r.target == "OpenAI"

    def test_extraction_result(self, mock_entities, mock_relations):
        from rag.graph.extractor import ExtractionResult
        result = ExtractionResult(entities=mock_entities, relations=mock_relations)
        assert len(result.entities) == 5
        assert len(result.relations) == 4


class TestGraphStore:
    """NetworkX 图构建 + PageRank (不需要 ES)"""

    def _make_store(self):
        from rag.graph.graph_store import GraphStore
        store = GraphStore.__new__(GraphStore)
        store.graph = nx.DiGraph()
        store._entity_map = {}
        store.es_conn = None
        store.emb_mdl = None
        return store

    def test_build_graph(self, mock_extraction):
        store = self._make_store()
        store.build_graph(mock_extraction)
        assert store.graph.number_of_nodes() == 5
        assert store.graph.number_of_edges() == 4

    def test_pagerank(self, mock_extraction):
        store = self._make_store()
        store.build_graph(mock_extraction)
        pr = store.compute_pagerank()
        assert len(pr) == 5
        assert all(0.0 <= v <= 1.0 for v in pr.values())

    def test_neighbors(self, mock_extraction):
        store = self._make_store()
        store.build_graph(mock_extraction)
        store.compute_pagerank()
        neighbors = store.get_neighbors("ChatGPT", n_hops=2)
        assert len(neighbors) > 0
        assert all("name" in nb for nb in neighbors)

    def test_save_load(self, mock_extraction):
        store = self._make_store()
        store.build_graph(mock_extraction)
        store.compute_pagerank()

        tmp = tempfile.mktemp(suffix=".json")
        try:
            store.save_graph(tmp)
            assert os.path.exists(tmp)

            store2 = self._make_store()
            store2.load_graph(tmp)
            assert store2.graph.number_of_nodes() == 5
            assert store2.graph.number_of_edges() == 4
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_empty_graph_pagerank(self):
        store = self._make_store()
        pr = store.compute_pagerank()
        assert pr == {}


class TestGraphSearchFormat:
    """图谱检索结果格式化测试"""

    def test_format_context(self):
        from rag.graph.graph_search import GraphSearcher
        entities = [
            {"name": "微软", "type": "ORGANIZATION", "description": "科技公司",
             "pagerank": 0.85, "fusion_score": 0.9},
        ]
        relations = [
            {"source": "微软", "target": "OpenAI", "description": "投资了",
             "pagerank": 0.85, "fusion_score": 0.9},
        ]
        paths = [
            {"from": "ChatGPT", "to": "OpenAI", "type": "ORGANIZATION",
             "relation": "开发了", "hop": 1, "pagerank": 0.72},
        ]
        context = GraphSearcher.format_context(entities, relations, paths)
        assert "知识图谱上下文" in context
        assert "微软" in context

    def test_format_empty(self):
        from rag.graph.graph_search import GraphSearcher
        context = GraphSearcher.format_context([], [], [])
        assert isinstance(context, str)


class TestAbstractInterfaces:
    """抽象接口合规性测试"""

    def test_chat_client_is_abc(self):
        from rag.llm.base import BaseChatClient
        from rag.llm.chat import ChatClient
        assert issubclass(ChatClient, BaseChatClient)

    def test_embedding_is_abc(self):
        from rag.llm.base import BaseEmbedding
        from rag.llm.embedding import RemoteEmbedding
        assert issubclass(RemoteEmbedding, BaseEmbedding)

    def test_reranker_is_abc(self):
        from rag.llm.base import BaseReranker
        from rag.llm.reranker import RemoteReranker
        assert issubclass(RemoteReranker, BaseReranker)

    def test_doc_store_is_abc(self):
        from rag.utils.doc_store_conn import DocStoreConnection
        from rag.utils.es_conn import ESConnection
        assert issubclass(ESConnection, DocStoreConnection)


class TestConfig:
    """配置加载测试"""

    def test_config_loads(self):
        from rag.settings import get_config, get_es_config, get_embedding_config, get_rag_config
        config = get_config()
        assert config is not None
        assert "es" in config
        assert "embedding" in config

    def test_es_config(self):
        from rag.settings import get_es_config
        es_cfg = get_es_config()
        assert "hosts" in es_cfg

    def test_embedding_config(self):
        from rag.settings import get_embedding_config
        emb_cfg = get_embedding_config()
        assert "model_name" in emb_cfg

    def test_rag_config_defaults(self):
        from rag.settings import get_rag_config
        rag_cfg = get_rag_config()
        assert rag_cfg["chunk_token_num"] == 512
