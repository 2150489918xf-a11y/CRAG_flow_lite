"""
LLM 抽象基类 + 工厂函数
所有上层模块（graph, crag, query_enhance, deps）只依赖此抽象接口,
具体的 LLM 后端（OpenAI, Anthropic, Gemini, 本地模型等）作为可插拔实现注入。
"""
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
#  Chat 抽象
# ══════════════════════════════════════════

class BaseChatClient(ABC):
    """LLM Chat 统一接口"""

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.1, max_tokens: int = 4096) -> str:
        """同步 Chat，返回纯文本"""
        ...

    @abstractmethod
    async def achat(self, system_prompt: str, user_prompt: str,
                    temperature: float = 0.1, max_tokens: int = 4096) -> str:
        """异步 Chat"""
        ...

    @abstractmethod
    def chat_json(self, system_prompt: str, user_prompt: str,
                  temperature: float = 0.1, max_tokens: int = 4096) -> dict:
        """同步 Chat，返回解析后的 JSON"""
        ...

    @abstractmethod
    async def achat_json(self, system_prompt: str, user_prompt: str,
                         temperature: float = 0.1, max_tokens: int = 4096) -> dict:
        """异步 Chat JSON"""
        ...


# ══════════════════════════════════════════
#  Embedding 抽象
# ══════════════════════════════════════════

class BaseEmbedding(ABC):
    """Embedding 模型统一接口"""

    @abstractmethod
    def encode(self, texts: list) -> tuple:
        """
        批量编码文本 → 向量。
        Returns: (np.ndarray, token_count: int)
        """
        ...

    @abstractmethod
    def encode_queries(self, text: str) -> tuple:
        """
        单条查询编码。
        Returns: (np.ndarray, token_count: int)
        """
        ...


# ══════════════════════════════════════════
#  Reranker 抽象
# ══════════════════════════════════════════

class BaseReranker(ABC):
    """Reranker 模型统一接口"""

    @abstractmethod
    def rerank(self, query: str, documents: list[str],
               top_n: int = None, return_documents: bool = False) -> list[dict]:
        """
        对候选文档重排序。
        Returns: [{"index": int, "relevance_score": float}, ...]
        """
        ...

    @abstractmethod
    def rerank_chunks(self, query: str, chunks: list[dict],
                      text_field: str = "content_with_weight",
                      top_n: int = None) -> list[dict]:
        """
        对 chunk 字典列表做重排序。
        Returns: 重排后的 chunk 列表
        """
        ...


# ══════════════════════════════════════════
#  工厂函数 (注册器驱动)
#
#  切换后端只需:
#    1. 实现 ABC 子类并 @registry.register("backend_name")
#    2. 在 service_conf.yaml 设置 backend: backend_name
#
#  例:
#    llm:
#      backend: anthropic   # 切换 Chat 后端
#    embedding:
#      backend: local       # 切换 Embedding 后端
# ══════════════════════════════════════════

_chat_instance: Optional[BaseChatClient] = None
_emb_instance: Optional[BaseEmbedding] = None
_reranker_instance: Optional[BaseReranker] = None


def get_chat_client(**kwargs) -> BaseChatClient:
    """
    获取 Chat 客户端单例。
    通过 service_conf.yaml 的 llm.backend 配置切换后端 (默认 "openai")。
    """
    global _chat_instance
    if _chat_instance is not None:
        return _chat_instance

    from common.registry import chat_registry
    from rag.settings import get_config

    # 确保内置后端已注册
    import rag.llm.chat  # noqa: F401

    cfg = get_config().get("llm", {})
    backend = cfg.get("backend", "openai")

    _chat_instance = chat_registry.create(backend, **kwargs)
    logger.info(f"ChatClient initialized: backend={backend}")
    return _chat_instance


def get_embedding(**kwargs) -> BaseEmbedding:
    """
    获取 Embedding 模型单例。
    通过 service_conf.yaml 的 embedding.backend 配置切换后端 (默认 "openai")。
    """
    global _emb_instance
    if _emb_instance is not None:
        return _emb_instance

    from common.registry import embedding_registry
    from rag.settings import get_embedding_config

    # 确保内置后端已注册
    import rag.llm.embedding  # noqa: F401

    cfg = get_embedding_config()
    backend = cfg.get("backend", "openai")

    _emb_instance = embedding_registry.create(
        backend,
        api_key=kwargs.get("api_key", cfg.get("api_key", "")),
        model_name=kwargs.get("model_name", cfg.get("model_name", "")),
        base_url=kwargs.get("base_url", cfg.get("base_url", "")),
    )
    logger.info(f"Embedding initialized: backend={backend}, model={getattr(_emb_instance, 'model_name', 'N/A')}")
    return _emb_instance


def get_reranker(**kwargs) -> Optional[BaseReranker]:
    """
    获取 Reranker 模型单例。
    通过 service_conf.yaml 的 reranker.backend 配置切换后端 (默认 "remote")。
    如未启用则返回 None。
    """
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance

    from common.registry import reranker_registry
    from rag.settings import get_config

    # 确保内置后端已注册
    import rag.llm.reranker  # noqa: F401

    cfg = get_config().get("reranker", {})
    if not cfg.get("enabled", False):
        return None

    backend = cfg.get("backend", "remote")

    _reranker_instance = reranker_registry.create(
        backend,
        api_key=kwargs.get("api_key", cfg.get("api_key", "")),
        model_name=kwargs.get("model_name", cfg.get("model_name", "BAAI/bge-reranker-v2-m3")),
        base_url=kwargs.get("base_url", cfg.get("base_url", "")),
    )
    logger.info(f"Reranker initialized: backend={backend}, model={getattr(_reranker_instance, 'model_name', 'N/A')}")
    return _reranker_instance
