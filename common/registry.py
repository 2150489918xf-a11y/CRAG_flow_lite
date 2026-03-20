"""
统一插件注册器
用法:
    # 1. 创建注册器
    doc_store_registry = Registry("doc_store")

    # 2. 注册实现
    @doc_store_registry.register("elasticsearch")
    class ESConnection(DocStoreConnection):
        ...

    # 3. 工厂创建
    conn = doc_store_registry.create("elasticsearch", hosts="...")

    # 4. 列出已注册
    doc_store_registry.list()  # ["elasticsearch"]
"""
import logging
from typing import Any, Dict, Type

logger = logging.getLogger(__name__)


class Registry:
    """
    通用组件注册器

    支持:
    - @registry.register("name") 装饰器注册
    - registry.create("name", **kwargs) 工厂创建
    - registry.list() 列出所有已注册实现
    - registry.get("name") 获取类（不实例化）
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: str):
        """装饰器：注册一个实现类"""
        def decorator(cls):
            if key in self._registry:
                logger.warning(
                    f"Registry[{self.name}]: overwriting '{key}' "
                    f"({self._registry[key].__name__} → {cls.__name__})"
                )
            self._registry[key] = cls
            logger.debug(f"Registry[{self.name}]: registered '{key}' → {cls.__name__}")
            return cls
        return decorator

    def get(self, key: str) -> Type:
        """获取已注册的类（不实例化）"""
        if key not in self._registry:
            available = ", ".join(self._registry.keys()) or "(empty)"
            raise KeyError(
                f"Registry[{self.name}]: '{key}' not found. "
                f"Available: [{available}]"
            )
        return self._registry[key]

    def create(self, key: str, *args, **kwargs) -> Any:
        """工厂方法：创建已注册类的实例"""
        cls = self.get(key)
        return cls(*args, **kwargs)

    def list(self) -> list[str]:
        """列出所有已注册的 key"""
        return list(self._registry.keys())

    def has(self, key: str) -> bool:
        """检查某个 key 是否已注册"""
        return key in self._registry

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __repr__(self) -> str:
        items = ", ".join(self._registry.keys())
        return f"Registry('{self.name}', [{items}])"


# ══════════════════════════════════════════
#  全局注册器实例
# ══════════════════════════════════════════

doc_store_registry = Registry("doc_store")
chat_registry = Registry("chat")
embedding_registry = Registry("embedding")
reranker_registry = Registry("reranker")
parser_registry = Registry("parser")
chunker_registry = Registry("chunker")
