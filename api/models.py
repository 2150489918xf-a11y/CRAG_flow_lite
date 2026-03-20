"""
API 请求/响应数据模型
"""
from typing import Optional
from pydantic import BaseModel


class KnowledgeBaseCreate(BaseModel):
    kb_id: str
    description: str = ""


class BatchDeleteRequest(BaseModel):
    kb_ids: list[str]


class RetrievalRequest(BaseModel):
    question: str
    kb_ids: list[str]
    top_k: int = 5
    similarity_threshold: float = 0.1
    vector_similarity_weight: float = 0.3
    highlight: bool = False


class GraphRetrievalRequest(BaseModel):
    """GraphRAG + CRAG 增强检索请求"""
    question: str
    kb_ids: list[str]
    top_k: int = 5
    similarity_threshold: float = 0.1
    vector_similarity_weight: float = 0.3
    highlight: bool = False
    enable_graph: bool = True
    enable_crag: bool = True
    n_hops: int = 2
    max_entities: int = 10
    max_relations: int = 15


class ChunkResponse(BaseModel):
    chunk_id: str
    content: str
    doc_name: str
    similarity: float
    vector_similarity: float = 0.0
    term_similarity: float = 0.0


class RetrievalResponse(BaseModel):
    total: int
    chunks: list[dict]
    doc_aggs: list[dict]


class GraphRetrievalResponse(BaseModel):
    """GraphRAG + CRAG 增强检索响应"""
    total: int
    chunks: list[dict]
    doc_aggs: list[dict]
    graph_entities: list[dict] = []
    graph_relations: list[dict] = []
    graph_paths: list[dict] = []
    graph_context: str = ""
    crag_score: str = ""
    crag_reason: str = ""
    crag_action: str = ""
    crag_latency_ms: int = 0
