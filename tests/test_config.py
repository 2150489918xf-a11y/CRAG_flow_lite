"""
配置 Schema 校验测试：验证 Pydantic 模型对合法/非法配置的行为
"""
import pytest
from pydantic import ValidationError


class TestConfigSchemaValid:
    """合法配置校验"""

    def test_load_current_config(self):
        """当前 service_conf.yaml 应能通过校验"""
        from rag.settings import get_service_config
        cfg = get_service_config()
        assert cfg.es.hosts == "http://localhost:9200"
        assert cfg.embedding.model_name == "BAAI/bge-m3"
        assert cfg.rag.chunk_token_num == 512

    def test_backward_compatible_api(self):
        """get_config() 仍返回 dict"""
        from rag.settings import get_config
        cfg = get_config()
        assert isinstance(cfg, dict)
        assert "es" in cfg

    def test_es_config_dict(self):
        from rag.settings import get_es_config
        es = get_es_config()
        assert isinstance(es, dict)
        assert "hosts" in es

    def test_embedding_config_dict(self):
        from rag.settings import get_embedding_config
        emb = get_embedding_config()
        assert isinstance(emb, dict)
        assert "model_name" in emb
        assert "api_key" in emb

    def test_rag_config_defaults(self):
        from rag.settings import get_rag_config
        rag = get_rag_config()
        assert rag["chunk_token_num"] == 512
        assert rag["top_k"] == 5
        assert isinstance(rag["use_parent_child"], bool)


class TestConfigSchemaInvalid:
    """非法配置校验 — 确保 Pydantic 能正确拒绝"""

    def test_missing_embedding_api_key(self):
        from rag.settings import ServiceConfig
        with pytest.raises(ValidationError, match="embedding"):
            ServiceConfig(embedding={"model_name": "test"})

    def test_missing_embedding_model_name(self):
        from rag.settings import ServiceConfig
        with pytest.raises(ValidationError, match="embedding"):
            ServiceConfig(embedding={"api_key": "test"})

    def test_port_out_of_range(self):
        from rag.settings import ServerConfig
        with pytest.raises(ValidationError):
            ServerConfig(port=99999)

    def test_port_negative(self):
        from rag.settings import ServerConfig
        with pytest.raises(ValidationError):
            ServerConfig(port=-1)

    def test_chunk_token_too_small(self):
        from rag.settings import RAGConfig
        with pytest.raises(ValidationError):
            RAGConfig(chunk_token_num=10)

    def test_similarity_threshold_out_of_range(self):
        from rag.settings import RAGConfig
        with pytest.raises(ValidationError):
            RAGConfig(similarity_threshold=1.5)

    def test_child_bigger_than_parent(self):
        from rag.settings import RAGConfig
        with pytest.raises(ValidationError, match="child_token_num"):
            RAGConfig(use_parent_child=True, parent_token_num=256, child_token_num=512)

    def test_unknown_field_rejected(self):
        """拼写错误的字段应被拒绝"""
        from rag.settings import ServiceConfig
        with pytest.raises(ValidationError, match="extra"):
            ServiceConfig(
                embedding={"api_key": "x", "model_name": "y"},
                unknwon_field="oops"
            )

    def test_top_k_zero(self):
        from rag.settings import RAGConfig
        with pytest.raises(ValidationError):
            RAGConfig(top_k=0)

    def test_graph_n_hops_too_large(self):
        from rag.settings import GraphConfig
        with pytest.raises(ValidationError):
            GraphConfig(n_hops=10)
