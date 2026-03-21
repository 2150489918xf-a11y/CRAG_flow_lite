"""
Agent Tool API — 供外部 Agent 调用的检索工具接口

职责：纯检索工具，不做 LLM 生成。返回拼装好的上下文 + 溯源，Agent 侧自行生成答案。

端点:
  POST /api/tool/retrieve  — 核心检索 (fast/hybrid/deep 三种模式)
  GET  /api/tool/list_kbs   — 列出可用知识库
  GET  /api/tool/schema     — 返回 OpenAI Function Calling JSON Schema
"""
import logging
import time

from fastapi import APIRouter

from api.deps import (
    get_dealer, get_emb, get_reranker, get_graph_searcher,
    get_crag_router, get_query_enhancer, get_es,
)
from api.models import (
    ToolRetrieveRequest, ToolRetrieveResponse,
    ToolSource, ToolMetadata,
)
from api.errors import ValidationError, ok_response
from rag.nlp.search import index_name

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tool", tags=["Agent Tool"])


# ══════════════════════════════════════════
#  OpenAI Function Calling Schema
# ══════════════════════════════════════════

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_retrieve",
        "description": (
            "从 RAGFlow Lite 知识库中检索与用户问题相关的文档片段。"
            "返回拼装好的上下文文本和溯源列表，可直接用于生成答案。"
            "支持三种模式：fast(低延迟)、hybrid(含图谱推理)、deep(含CRAG纠错)。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户的问题或检索查询",
                },
                "kb_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要搜索的知识库 ID 列表，留空则搜索全部知识库",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的最相关文档片段数量",
                    "default": 5,
                },
                "mode": {
                    "type": "string",
                    "enum": ["fast", "hybrid", "deep"],
                    "description": "检索模式：fast=低延迟, hybrid=含图谱推理, deep=含CRAG纠错",
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        },
    },
}


# ══════════════════════════════════════════
#  端点实现
# ══════════════════════════════════════════

@router.get("/schema")
async def get_tool_schema():
    """返回 OpenAI Function Calling 兼容的 JSON Schema"""
    return ok_response(TOOL_SCHEMA)


@router.get("/list_kbs")
async def list_kbs():
    """列出所有可用知识库（供 Agent 发现）"""
    es = get_es()
    try:
        indices = es.es.indices.get(index="ragflow_lite_*")
        kbs = []
        for idx_name_str, info in indices.items():
            kb_id = idx_name_str.replace("ragflow_lite_", "")
            count = es.es.count(index=idx_name_str)["count"]
            meta = es.get_index_meta(idx_name_str)
            kbs.append({
                "kb_id": kb_id,
                "display_name": meta.get("display_name", kb_id),
                "doc_count": count,
            })
        return ok_response({"knowledgebases": kbs})
    except Exception:
        return ok_response({"knowledgebases": []})


@router.post("/retrieve", response_model=ToolRetrieveResponse)
async def tool_retrieve(req: ToolRetrieveRequest):
    """
    Agent 检索工具 — 核心端点

    根据 mode 自动选择检索深度:
      fast   → ES 混合检索 + Reranker
      hybrid → fast + GraphRAG 图谱推理
      deep   → hybrid + CRAG 纠错路由
    """
    t0 = time.perf_counter()

    if not req.query.strip():
        raise ValidationError("query 不能为空")

    mode = req.mode.lower()
    if mode not in ("fast", "hybrid", "deep"):
        raise ValidationError(f"mode 必须为 fast/hybrid/deep，收到: {mode}")

    # ── 自动发现知识库 ──
    kb_ids = req.kb_ids
    if not kb_ids:
        es = get_es()
        try:
            indices = es.es.indices.get(index="ragflow_lite_*")
            kb_ids = [name.replace("ragflow_lite_", "") for name in indices.keys()]
        except Exception:
            kb_ids = []
    if not kb_ids:
        return ToolRetrieveResponse(
            answer_context="未找到任何知识库。",
            sources=[],
            metadata=ToolMetadata(mode=mode, latency_ms=0),
        )

    dealer = get_dealer()
    emb_mdl = get_emb()
    import asyncio

    # ── Step 1: ES 混合检索 ──
    es_result = await dealer.retrieval(
        question=req.query,
        embd_mdl=emb_mdl,
        kb_ids=kb_ids,
        page=1,
        page_size=req.top_k * 3,
        similarity_threshold=0.1,
        vector_similarity_weight=0.3,
        highlight=False,
        query_enhancer=get_query_enhancer(),
    )
    chunks = es_result.get("chunks", [])
    total = es_result.get("total", 0)

    # ── Step 2: Reranker 精排 ──
    reranker = get_reranker()
    if reranker and chunks:
        try:
            chunks = reranker.rerank_chunks(req.query, chunks, top_n=req.top_k)
        except Exception as e:
            logger.warning(f"Reranker failed: {e}")
            chunks = chunks[:req.top_k]
    else:
        chunks = chunks[:req.top_k]

    graph_context = ""
    crag_score = ""
    crag_reason = ""

    # ── Step 3: GraphRAG (hybrid / deep) ──
    if mode in ("hybrid", "deep"):
        gs = get_graph_searcher()
        if gs:
            try:
                qa = await gs.rewrite_query(req.query)
                if qa:
                    graph_result = await gs.search_with_qa(
                        question=req.query, kb_ids=kb_ids, qa=qa,
                        topk_entity=20, topk_relation=30, n_hops=2,
                    )
                    graph_context = graph_result.formatted_context
            except Exception as e:
                logger.warning(f"GraphRAG failed: {e}")

    # ── Step 4: CRAG (deep only) ──
    if mode == "deep":
        crag = get_crag_router()
        if crag:
            try:
                crag_result = await crag.route(
                    question=req.query,
                    local_chunks=chunks,
                    graph_context=graph_context,
                )
                chunks = crag_result["chunks"]
                graph_context = crag_result["graph_context"]
                crag_score = crag_result["crag_score"]
                crag_reason = crag_result["crag_reason"]
            except Exception as e:
                logger.warning(f"CRAG failed: {e}")

    # ── Step 5: 组装 answer_context + sources ──
    context_parts = []
    sources = []

    # 图谱上下文
    if graph_context.strip():
        context_parts.append(f"=== 知识图谱推理 ===\n{graph_context}")
        sources.append(ToolSource(
            id="graph_context",
            content=graph_context[:200] + "..." if len(graph_context) > 200 else graph_context,
            doc_name="[知识图谱]",
            source_type="graph",
            relevance_score=1.0,
        ))

    # 文本 chunks
    context_parts.append("=== 检索文档片段 ===")
    for i, c in enumerate(chunks):
        content = c.get("content_with_weight", "")
        doc_name = c.get("docnm_kwd", "")
        chunk_id = c.get("chunk_id", c.get("id", f"chunk_{i}"))
        similarity = c.get("similarity", 0.0)
        source_type = "web" if c.get("knowledge_graph_kwd") == "web" else "local"

        context_parts.append(f"[{i+1}] ({doc_name}) {content}")
        sources.append(ToolSource(
            id=chunk_id,
            content=content,
            doc_name=doc_name,
            source_type=source_type,
            relevance_score=round(similarity, 4) if isinstance(similarity, (int, float)) else 0.0,
        ))

    answer_context = "\n\n".join(context_parts)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return ToolRetrieveResponse(
        answer_context=answer_context,
        sources=sources,
        metadata=ToolMetadata(
            mode=mode,
            total_hits=total,
            source_count=len(sources),
            latency_ms=latency_ms,
            crag_score=crag_score,
            crag_reason=crag_reason,
        ),
    )
