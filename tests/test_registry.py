"""
Registry 单元测试
"""
import pytest
from common.registry import Registry


class TestRegistry:
    """测试通用插件注册器"""

    def setup_method(self):
        self.r = Registry("test")

    def test_register_and_create(self):
        @self.r.register("foo")
        class Foo:
            def __init__(self, x=1):
                self.x = x

        obj = self.r.create("foo", x=42)
        assert obj.x == 42

    def test_get_class(self):
        @self.r.register("bar")
        class Bar:
            pass

        cls = self.r.get("bar")
        assert cls is Bar

    def test_list(self):
        @self.r.register("a")
        class A: pass

        @self.r.register("b")
        class B: pass

        assert set(self.r.list()) == {"a", "b"}

    def test_has_and_contains(self):
        @self.r.register("x")
        class X: pass

        assert self.r.has("x")
        assert "x" in self.r
        assert not self.r.has("y")
        assert "y" not in self.r

    def test_missing_key_raises(self):
        with pytest.raises(KeyError, match="not found"):
            self.r.get("nonexistent")

    def test_create_missing_key_raises(self):
        with pytest.raises(KeyError, match="not found"):
            self.r.create("nonexistent")

    def test_overwrite_warning(self):
        @self.r.register("dup")
        class A: pass

        @self.r.register("dup")
        class B: pass

        # Last registration wins
        assert self.r.get("dup") is B

    def test_repr(self):
        @self.r.register("item")
        class Item: pass

        assert "test" in repr(self.r)
        assert "item" in repr(self.r)


class TestGlobalRegistries:
    """测试全局注册器实例存在且可用"""

    def test_registries_exist(self):
        from common.registry import (
            doc_store_registry, chat_registry, embedding_registry,
            reranker_registry, parser_registry, chunker_registry,
        )
        assert doc_store_registry.name == "doc_store"
        assert chat_registry.name == "chat"
        assert embedding_registry.name == "embedding"
        assert reranker_registry.name == "reranker"
        assert parser_registry.name == "parser"
        assert chunker_registry.name == "chunker"

    def test_es_registered(self):
        """验证 ES 后端已通过 @register 注册"""
        import rag.utils.es_conn  # noqa: trigger registration
        from common.registry import doc_store_registry
        assert "elasticsearch" in doc_store_registry

    def test_chat_registered(self):
        import rag.llm.chat  # noqa
        from common.registry import chat_registry
        assert "openai" in chat_registry

    def test_embedding_registered(self):
        import rag.llm.embedding  # noqa
        from common.registry import embedding_registry
        assert "openai" in embedding_registry

    def test_reranker_registered(self):
        import rag.llm.reranker  # noqa
        from common.registry import reranker_registry
        assert "remote" in reranker_registry
