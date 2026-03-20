"""
检索服务路由 (Retrieval + GraphRAG + CRAG)
"""
import asyncio
import logging

from fastapi import APIRouter

from api.deps import (
    get_dealer, get_emb, get_reranker, get_graph_searcher,
    get_crag_router, get_query_enhancer,
)
from api.models import (
    RetrievalRequest, RetrievalResponse,
    GraphRetrievalRequest, GraphRetrievalResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["检索服务"])


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieval(req: RetrievalRequest):
    """混合检索 + Reranker 精排"""
    dealer = get_dealer()
    emb_mdl = get_emb()

    result = await dealer.retrieval(
        question=req.question,
        embd_mdl=emb_mdl,
        kb_ids=req.kb_ids,
        page=1,
        page_size=req.top_k * 3,
        similarity_threshold=req.similarity_threshold,
        vector_similarity_weight=req.vector_similarity_weight,
        highlight=req.highlight,
        query_enhancer=get_query_enhancer(),
    )

    chunks = result.get("chunks", [])

    # Reranker 精排
    reranker = get_reranker()
    if reranker and chunks:
        try:
            chunks = reranker.rerank_chunks(
                req.question, chunks, top_n=req.top_k
            )
            logger.info(f"Reranked {len(result.get('chunks', []))} → {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"Reranker failed, using original order: {e}")
            chunks = chunks[:req.top_k]
    else:
        chunks = chunks[:req.top_k]

    return RetrievalResponse(
        total=result.get("total", 0),
        chunks=chunks,
        doc_aggs=result.get("doc_aggs", []),
    )


@router.post("/graph_retrieval", response_model=GraphRetrievalResponse)
async def graph_retrieval(req: GraphRetrievalRequest):
    """
    GraphRAG + CRAG 增强检索

    完整流程：
    1. 混合召回 → ES fulltext + KNN
    2. Reranker 精排 → BGE-Reranker
    3. GraphRAG 图谱检索 → 四路并行 + PageRank 融合
    4. CRAG 动态路由 → Correct/Incorrect/Ambiguous 三路流转
    5. Prompt 组装 → 图谱CSV + 纯净文本 chunks 返回
    """
    dealer = get_dealer()
    emb_mdl = get_emb()

    # ===== Step 1: ES 混合召回 + GraphRAG 查询改写 并行执行 =====
    es_task = dealer.retrieval(
        question=req.question,
        embd_mdl=emb_mdl,
        kb_ids=req.kb_ids,
        page=1,
        page_size=req.top_k * 3,
        similarity_threshold=req.similarity_threshold,
        vector_similarity_weight=req.vector_similarity_weight,
        highlight=req.highlight,
        query_enhancer=get_query_enhancer(),
    )

    gs = get_graph_searcher() if req.enable_graph else None
    rewrite_task = gs.rewrite_query(req.question) if gs else None

    if rewrite_task:
        text_result, qa = await asyncio.gather(es_task, rewrite_task)
    else:
        text_result = await es_task
        qa = None

    text_chunks = text_result.get("chunks", [])
    doc_aggs = text_result.get("doc_aggs", [])
    if not isinstance(doc_aggs, list):
        doc_aggs = list(doc_aggs.values()) if isinstance(doc_aggs, dict) else []

    # ===== Step 2: Reranker 精排 =====
    reranker = get_reranker()
    if reranker and text_chunks:
        try:
            text_chunks = reranker.rerank_chunks(
                req.question, text_chunks, top_n=req.top_k
            )
            logger.info(f"Reranked text chunks → {len(text_chunks)}")
        except Exception as e:
            logger.warning(f"Reranker failed: {e}")
            text_chunks = text_chunks[:req.top_k]
    else:
        text_chunks = text_chunks[:req.top_k]

    # ===== Step 3: GraphRAG 图谱检索 =====
    graph_entities = []
    graph_relations = []
    graph_paths = []
    graph_context = ""

    if req.enable_graph and gs and qa:
        try:
            graph_result = await gs.search_with_qa(
                question=req.question,
                kb_ids=req.kb_ids,
                qa=qa,
                topk_entity=req.max_entities * 2,
                topk_relation=req.max_relations * 2,
                n_hops=req.n_hops,
            )
            graph_entities = graph_result.entities[:req.max_entities]
            graph_relations = graph_result.relations[:req.max_relations]
            graph_paths = graph_result.paths
            graph_context = graph_result.formatted_context
        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")

    # ===== Step 4: CRAG 动态路由 =====
    crag_score = "disabled"
    crag_reason = ""
    crag_action = ""
    crag_latency = 0

    if req.enable_crag:
        crag = get_crag_router()
        if crag:
            try:
                crag_result = await crag.route(
                    question=req.question,
                    local_chunks=text_chunks,
                    graph_context=graph_context,
                )
                text_chunks = crag_result["chunks"]
                graph_context = crag_result["graph_context"]
                crag_score = crag_result["crag_score"]
                crag_reason = crag_result["crag_reason"]
                crag_action = crag_result["crag_action"]
                crag_latency = crag_result["latency_ms"]
                logger.info(f"CRAG: {crag_score} — {crag_action} ({crag_latency}ms)")
            except Exception as e:
                logger.error(f"CRAG failed, using original data: {e}")
                crag_score = "error"
                crag_reason = str(e)[:100]

    # ===== Step 5: 组装最终 chunks =====
    if graph_context.strip():
        graph_chunk = {
            "chunk_id": "graph_context",
            "content_with_weight": graph_context,
            "docnm_kwd": "[知识图谱]",
            "doc_type_kwd": "knowledge_graph",
            "similarity": 1.0,
        }
        text_chunks.insert(0, graph_chunk)

    return GraphRetrievalResponse(
        total=text_result.get("total", 0) + len(graph_entities),
        chunks=text_chunks,
        doc_aggs=doc_aggs,
        graph_entities=graph_entities,
        graph_relations=graph_relations,
        graph_paths=graph_paths,
        graph_context=graph_context,
        crag_score=crag_score,
        crag_reason=crag_reason,
        crag_action=crag_action,
        crag_latency_ms=crag_latency,
    )
