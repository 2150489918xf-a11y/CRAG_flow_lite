"""
RAGFlow Lite 配置加载器
"""
import os
import yaml

_config = None
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_base_directory():
    """返回项目根目录"""
    return _BASE_DIR


def get_config():
    """加载并缓存配置"""
    global _config
    if _config is not None:
        return _config

    conf_path = os.path.join(_BASE_DIR, "conf", "service_conf.yaml")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"配置文件不存在: {conf_path}")

    with open(conf_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    return _config


def get_es_config():
    """获取 ES 配置"""
    cfg = get_config()
    return cfg.get("es", {})


def get_embedding_config():
    """获取 Embedding 配置"""
    cfg = get_config()
    return cfg.get("embedding", {})


def get_rag_config():
    """获取 RAG 配置"""
    cfg = get_config()
    defaults = {
        "chunk_token_num": 512,
        "delimiter": "\n!?。；！？",
        "top_k": 5,
        "similarity_threshold": 0.2,
        "vector_similarity_weight": 0.3,
    }
    rag_cfg = cfg.get("rag", {})
    for k, v in defaults.items():
        if k not in rag_cfg:
            rag_cfg[k] = v
    return rag_cfg
