"""
检索服务路由 (Retrieval + GraphRAG + CRAG)
各阶段自动计时并记录到 PerfCollector
"""
import asyncio
import logging
import time

from fastapi import APIRouter

from api.deps import (
    get_dealer, get_emb, get_reranker, get_graph_searcher,
    get_crag_router, get_query_enhancer,
)
from api.models import (
    RetrievalRequest, RetrievalResponse,
    GraphRetrievalRequest, GraphRetrievalResponse,
)
from common.perf import perf

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["检索服务"])


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieval(req: RetrievalRequest):
    """混合检索 + Reranker 精排"""
    t_total = time.perf_counter()

    dealer = get_dealer()
    emb_mdl = get_emb()

    with perf.timer("es_retrieval"):
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
            with perf.timer("reranker"):
                chunks = reranker.rerank_chunks(
                    req.question, chunks, top_n=req.top_k
                )
            logger.info(f"Reranked {len(result.get('chunks', []))} → {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"Reranker failed, using original order: {e}")
            chunks = chunks[:req.top_k]
    else:
        chunks = chunks[:req.top_k]

    perf.record("total_retrieval", (time.perf_counter() - t_total) * 1000)

    return RetrievalResponse(
        total=result.get("total", 0),
        chunks=chunks,
        doc_aggs=result.get("doc_aggs", []),
    )


@router.post("/graph_retrieval", response_model=GraphRetrievalResponse)
async def graph_retrieval(req: GraphRetrievalRequest):
    """
    GraphRAG + CRAG 增强检索（含各阶段性能埋点）
    """
    t_total = time.perf_counter()
    dealer = get_dealer()
    emb_mdl = get_emb()

    # ===== Step 1: ES 混合召回 + GraphRAG 查询改写 并行执行 =====
    with perf.timer("es_retrieval"):
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
            with perf.timer("reranker"):
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
            with perf.timer("graph_search"):
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
                with perf.timer("crag_routing"):
                    crag_result = await crag.route(
                        question=req.question,
                        local_chunks=text_chunks,
                        graph_context=graph_context,
                        enable_web_search=req.enable_web_search,
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

    elif req.enable_web_search:
        # CRAG 未开启，但用户手动开启了网络检索 → 直接搜索并追加到结果
        from rag.crag.web_search import WebSearcher
        try:
            t_ws = time.perf_counter()
            web_searcher = WebSearcher()
            web_chunks = await web_searcher.search(req.question, top_k=3)
            ws_latency = int((time.perf_counter() - t_ws) * 1000)
            if web_chunks:
                text_chunks = text_chunks + web_chunks
                crag_score = "web_only"
                crag_action = f"WEB_SEARCH_DIRECT: 直接外网检索，召回 {len(web_chunks)} 条"
                crag_reason = "CRAG 未开启，直接执行网络检索"
                crag_latency = ws_latency
                logger.info(f"Direct web search: {len(web_chunks)} results ({ws_latency}ms)")
            else:
                crag_score = "web_only"
                crag_action = "WEB_SEARCH_DIRECT_EMPTY: 外网检索无结果"
                crag_reason = "网络检索未返回结果"
                crag_latency = ws_latency
        except Exception as e:
            logger.error(f"Direct web search failed: {e}")
            crag_score = "error"
            crag_reason = f"网络检索失败: {str(e)[:80]}"

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

    perf.record("total_graph_retrieval", (time.perf_counter() - t_total) * 1000)

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
