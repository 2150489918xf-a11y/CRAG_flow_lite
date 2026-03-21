"""
RAGFlow Lite 配置加载器 — Pydantic Schema 校验版
配置文件: conf/service_conf.yaml
配错 key / 类型不对 / 缺少必填字段时，启动即刻报错并给出清晰提示。
"""
import os
import logging
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from common.paths import PROJECT_ROOT, CONF_DIR

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
#  各段 Schema
# ══════════════════════════════════════════

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=9380, ge=1, le=65535)


class ESConfig(BaseModel):
    hosts: str = "http://localhost:9200"
    username: str = ""
    password: str = ""


class EmbeddingConfig(BaseModel):
    api_key: str = Field(..., min_length=1, description="Embedding API Key (必填)")
    model_name: str = Field(..., min_length=1, description="Embedding 模型名称 (必填)")
    base_url: str = "https://api.openai.com/v1"


class LLMConfig(BaseModel):
    api_key: str = Field(default="", description="LLM API Key")
    model_name: str = Field(default="", description="LLM 模型名称")
    base_url: str = "https://api.openai.com/v1"


class RAGConfig(BaseModel):
    chunk_token_num: int = Field(default=512, ge=64, le=8192, description="分块 token 数")
    delimiter: str = "\n!?。；！？"
    top_k: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    vector_similarity_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    use_parent_child: bool = False
    parent_token_num: int = Field(default=1024, ge=128, le=16384)
    child_token_num: int = Field(default=256, ge=32, le=4096)

    @model_validator(mode="after")
    def validate_parent_child(self):
        if self.use_parent_child and self.child_token_num >= self.parent_token_num:
            raise ValueError(
                f"child_token_num ({self.child_token_num}) 必须小于 "
                f"parent_token_num ({self.parent_token_num})"
            )
        return self


class RerankerConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    model_name: str = "BAAI/bge-reranker-v2-m3"
    base_url: str = "https://api.siliconflow.cn/v1"
    top_n: int = Field(default=5, ge=1, le=100)


class GraphConfig(BaseModel):
    enabled: bool = False
    n_hops: int = Field(default=2, ge=1, le=5)
    max_entities: int = Field(default=10, ge=1, le=100)
    max_relations: int = Field(default=15, ge=1, le=200)


class CRAGConfig(BaseModel):
    enabled: bool = False
    tavily_api_key: str = ""
    search_provider: str = "tavily"


# ══════════════════════════════════════════
#  顶层配置 Schema
# ══════════════════════════════════════════

class ServiceConfig(BaseModel):
    """service_conf.yaml 完整 Schema"""
    server: ServerConfig = ServerConfig()
    es: ESConfig = ESConfig()
    embedding: EmbeddingConfig
    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()
    reranker: RerankerConfig = RerankerConfig()
    graph: GraphConfig = GraphConfig()
    crag: CRAGConfig = CRAGConfig()

    model_config = {"extra": "forbid"}  # 禁止未知字段，防止拼写错误


# ══════════════════════════════════════════
#  加载 & 缓存
# ══════════════════════════════════════════

_service_config: Optional[ServiceConfig] = None
_raw_config: Optional[dict] = None


def _load_and_validate() -> ServiceConfig:
    """加载 YAML 并用 Pydantic 校验"""
    global _service_config, _raw_config

    if _service_config is not None:
        return _service_config

    conf_path = CONF_DIR / "service_conf.yaml"
    if not conf_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {conf_path}")

    with open(conf_path, "r", encoding="utf-8") as f:
        _raw_config = yaml.safe_load(f) or {}

    try:
        _service_config = ServiceConfig(**_raw_config)
    except Exception as e:
        logger.error(f"配置校验失败: {e}")
        raise SystemExit(f"\n❌ 配置校验失败 ({conf_path}):\n{e}\n") from e

    logger.info(f"配置校验通过: ES={_service_config.es.hosts}, "
                f"Embedding={_service_config.embedding.model_name}")
    return _service_config


# ══════════════════════════════════════════
#  公共 API（保持与旧版完全兼容）
# ══════════════════════════════════════════

def get_project_base_directory():
    """返回项目根目录"""
    return str(PROJECT_ROOT)


def get_config() -> dict:
    """获取原始 dict 配置（向后兼容）"""
    _load_and_validate()
    return _raw_config


def get_service_config() -> ServiceConfig:
    """获取类型安全的配置对象"""
    return _load_and_validate()


def get_es_config() -> dict:
    """获取 ES 配置"""
    cfg = _load_and_validate()
    return cfg.es.model_dump()


def get_embedding_config() -> dict:
    """获取 Embedding 配置"""
    cfg = _load_and_validate()
    return cfg.embedding.model_dump()


def get_rag_config() -> dict:
    """获取 RAG 配置"""
    cfg = _load_and_validate()
    return cfg.rag.model_dump()
