"""
API 请求/响应数据模型
"""
from typing import Optional
from pydantic import BaseModel


class KnowledgeBaseCreate(BaseModel):
    kb_id: str
    description: str = ""
    folder: str = "/"


class BatchDeleteRequest(BaseModel):
    kb_ids: list[str]


class KBMoveRequest(BaseModel):
    """移动知识库到目标文件夹"""
    kb_id: str
    target_folder: str


class FolderCreateRequest(BaseModel):
    """创建文件夹"""
    path: str  # e.g. "/财务/政策法规"


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
    enable_web_search: bool = False
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


# ══════════════════════════════════════════
#  Agent Tool API 模型
# ══════════════════════════════════════════

class ToolRetrieveRequest(BaseModel):
    """
    Agent 检索工具请求 — 极简入参

    mode:
      - "fast":   ES 混合检索 + Reranker (低延迟)
      - "hybrid": fast + GraphRAG 图谱推理 (默认)
      - "deep":   hybrid + CRAG 纠错路由 (最准但慢)
    """
    query: str
    kb_ids: list[str] = []
    top_k: int = 5
    mode: str = "hybrid"
    folder: str = ""  # 按文件夹过滤 (如 "/财务")，留空搜全部
    enable_web_search: bool = False  # 是否启用网络检索


class ToolSource(BaseModel):
    """检索来源溯源"""
    id: str
    content: str
    doc_name: str = ""
    source_type: str = "local"  # "local" | "graph" | "web"
    relevance_score: float = 0.0


class ToolMetadata(BaseModel):
    """检索元信息"""
    mode: str
    total_hits: int = 0
    source_count: int = 0
    latency_ms: int = 0
    crag_score: str = ""
    crag_reason: str = ""
    crag_action: str = ""


class ToolRetrieveResponse(BaseModel):
    """
    Agent 检索工具响应

    answer_context: 拼装好的上下文文本，Agent 直接塞进 Prompt 即可
    sources:        溯源列表，供 Agent 引用
    metadata:       检索元信息
    """
    answer_context: str
    sources: list[ToolSource]
    metadata: ToolMetadata
