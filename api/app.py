"""
RAGFlow Lite FastAPI 应用
提供知识库管理、文档上传、检索三组 API
"""
import asyncio
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.app.chunking import chunk
from rag.llm.embedding import RemoteEmbedding
from rag.utils.es_conn import ESConnection
from rag.nlp.search import Dealer, index_name
from rag.settings import get_embedding_config, get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="RAGFlow Lite", version="0.3.0",
              description="轻量化 RAG 检索服务 (混合检索 + GraphRAG + CRAG)")

# 挂载静态前端文件
_static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir, html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局单例
_es_conn = None
_emb_mdl = None
_dealer = None
_graph_searcher = None
_reranker = None
_query_enhancer = None


def get_es():
    global _es_conn
    if _es_conn is None:
        _es_conn = ESConnection()
    return _es_conn


def get_emb():
    global _emb_mdl
    if _emb_mdl is None:
        cfg = get_embedding_config()
        _emb_mdl = RemoteEmbedding(
            api_key=cfg["api_key"],
            model_name=cfg["model_name"],
            base_url=cfg.get("base_url", ""),
        )
    return _emb_mdl


def get_dealer():
    global _dealer
    if _dealer is None:
        _dealer = Dealer(get_es())
    return _dealer


def get_graph_searcher():
    global _graph_searcher
    if _graph_searcher is None:
        cfg = get_config()
        graph_cfg = cfg.get("graph", {})
        if not graph_cfg.get("enabled", False):
            return None
        from rag.llm.chat import ChatClient
        from rag.graph.graph_search import GraphSearcher
        from rag.graph.graph_store import GraphStore
        chat = ChatClient()
        graph_store = GraphStore(es_conn=get_es(), emb_mdl=get_emb())

        # 尝试加载已有的图文件
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "graphs")
        if os.path.isdir(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith("_graph.json"):
                    graph_store.load_graph(os.path.join(data_dir, f))

        _graph_searcher = GraphSearcher(
            es_conn=get_es(), emb_mdl=get_emb(),
            chat_client=chat, graph_store=graph_store,
        )
    return _graph_searcher


def get_reranker():
    global _reranker
    if _reranker is None:
        cfg = get_config()
        reranker_cfg = cfg.get("reranker", {})
        if not reranker_cfg.get("enabled", False):
            return None
        from rag.llm.reranker import RemoteReranker
        _reranker = RemoteReranker(
            api_key=reranker_cfg.get("api_key", ""),
            model_name=reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"),
            base_url=reranker_cfg.get("base_url", "https://api.siliconflow.cn/v1"),
        )
    return _reranker


_crag_router = None

def get_crag_router():
    global _crag_router
    if _crag_router is None:
        cfg = get_config()
        crag_cfg = cfg.get("crag", {})
        if not crag_cfg.get("enabled", False):
            return None
        from rag.crag.router import CRAGRouter
        from rag.llm.chat import ChatClient
        _crag_router = CRAGRouter(chat_client=ChatClient())
    return _crag_router


def get_query_enhancer():
    global _query_enhancer
    if _query_enhancer is None:
        from rag.nlp.query_enhance import QueryEnhancer
        from rag.llm.chat import ChatClient
        _query_enhancer = QueryEnhancer(chat_client=ChatClient())
    return _query_enhancer


# ==================== 数据模型 ====================

class KnowledgeBaseCreate(BaseModel):
    kb_id: str
    description: str = ""


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
    crag_score: str = ""         # Correct / Incorrect / Ambiguous / disabled
    crag_reason: str = ""
    crag_action: str = ""
    crag_latency_ms: int = 0


# ==================== 知识库管理 ====================

@app.get("/")
async def root():
    """根路径重定向到前端页面"""
    return RedirectResponse(url="/static/index.html")


@app.get("/api/health")
async def health():
    try:
        info = get_es().health()
        cfg = get_config()
        return {
            "status": "ok",
            "es_status": info.get("status", "unknown"),
            "graph_enabled": cfg.get("graph", {}).get("enabled", False),
            "crag_enabled": cfg.get("crag", {}).get("enabled", False),
            "reranker_enabled": cfg.get("reranker", {}).get("enabled", False),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}



@app.post("/api/knowledgebase")
async def create_knowledgebase(req: KnowledgeBaseCreate):
    """创建知识库（ES 索引），支持中文名"""
    import re
    import uuid

    display_name = req.kb_id.strip()
    if not display_name:
        return {"status": "error", "detail": "知识库名称不能为空"}

    # Try to generate a readable safe_id from the input; fall back to UUID for non-ASCII names
    safe_id = re.sub(r'[^a-z0-9_]', '', req.kb_id.lower().replace(' ', '_').replace('-', '_'))
    if not safe_id:
        # Pure Chinese / non-ASCII name → generate short UUID-based ID
        safe_id = "kb_" + uuid.uuid4().hex[:8]

    es = get_es()
    idx = index_name(safe_id)

    # Check if an index with this display name already exists
    try:
        existing = es.es.indices.get(index="ragflow_lite_*")
        for existing_idx in existing:
            meta = es.get_index_meta(existing_idx)
            if meta.get("display_name") == display_name:
                existing_kb_id = existing_idx.replace("ragflow_lite_", "")
                return {"status": "exists", "kb_id": existing_kb_id, "display_name": display_name, "index": existing_idx}
    except Exception:
        pass

    if es.index_exist(idx):
        return {"status": "exists", "kb_id": safe_id, "display_name": display_name, "index": idx}

    es.create_idx(idx, display_name=display_name)
    return {"status": "created", "kb_id": safe_id, "display_name": display_name, "index": idx}


@app.get("/api/knowledgebase")
async def list_knowledgebases():
    """列出所有知识库"""
    es = get_es()
    try:
        indices = es.es.indices.get(index="ragflow_lite_*")
        kbs = []
        for idx_name_str, info in indices.items():
            kb_id = idx_name_str.replace("ragflow_lite_", "")
            count = es.es.count(index=idx_name_str)["count"]
            # Read display name from _meta
            meta = es.get_index_meta(idx_name_str)
            display_name = meta.get("display_name", kb_id)
            kbs.append({
                "kb_id": kb_id,
                "display_name": display_name,
                "index": idx_name_str,
                "doc_count": count,
            })
        return {"knowledgebases": kbs}
    except Exception as e:
        return {"knowledgebases": [], "error": str(e)}


@app.delete("/api/knowledgebase/{kb_id}")
async def delete_knowledgebase(kb_id: str):
    """删除知识库"""
    es = get_es()
    idx = index_name(kb_id)
    if es.delete_idx(idx):
        return {"status": "deleted", "kb_id": kb_id}
    raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


class BatchDeleteRequest(BaseModel):
    kb_ids: list[str]


@app.post("/api/knowledgebase/batch_delete")
async def batch_delete_knowledgebases(req: BatchDeleteRequest):
    """批量删除知识库"""
    if not req.kb_ids:
        return {"status": "error", "detail": "kb_ids 不能为空"}

    es = get_es()
    results = []
    deleted_count = 0
    failed_count = 0

    for kb_id in req.kb_ids:
        idx = index_name(kb_id)
        try:
            if es.delete_idx(idx):
                results.append({"kb_id": kb_id, "status": "deleted"})
                deleted_count += 1
            else:
                results.append({"kb_id": kb_id, "status": "not_found"})
                failed_count += 1
        except Exception as e:
            results.append({"kb_id": kb_id, "status": "error", "detail": str(e)})
            failed_count += 1
            logger.error(f"Failed to delete kb '{kb_id}': {e}")

    return {
        "status": "ok",
        "deleted": deleted_count,
        "failed": failed_count,
        "results": results,
    }


@app.get("/api/documents/{kb_id}")
async def list_documents(kb_id: str):
    """列出知识库中的所有文档及其分块数"""
    es = get_es()
    idx = index_name(kb_id)
    if not es.index_exist(idx):
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")

    try:
        r = es.es.search(
            index=idx,
            body={
                "size": 0,
                "query": {
                    "bool": {
                        "must_not": [
                            {"term": {"knowledge_graph_kwd": "entity"}},
                            {"term": {"knowledge_graph_kwd": "relation"}},
                        ]
                    }
                },
                "aggs": {
                    "docs": {
                        "terms": {"field": "docnm_kwd", "size": 500}
                    }
                },
            },
        )
        docs = []
        for bucket in r["aggregations"]["docs"]["buckets"]:
            docs.append({
                "doc_name": bucket["key"],
                "chunk_count": bucket["doc_count"],
            })
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        raise HTTPException(500, f"Failed to list documents: {e}")


@app.get("/api/chunks/{kb_id}")
async def list_chunks(kb_id: str, page: int = 1, page_size: int = 20,
                      doc_names: Optional[str] = None):
    """查看知识库的分块内容（分页，支持按文档筛选）"""
    es = get_es()
    idx = index_name(kb_id)
    if not es.index_exist(idx):
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")

    from_ = (page - 1) * page_size

    # 构建查询
    must_not = [
        {"term": {"knowledge_graph_kwd": "entity"}},
        {"term": {"knowledge_graph_kwd": "relation"}},
    ]
    must = []

    if doc_names:
        names = [n.strip() for n in doc_names.split(",") if n.strip()]
        if names:
            must.append({"terms": {"docnm_kwd": names}})

    query = {"bool": {"must_not": must_not}}
    if must:
        query["bool"]["must"] = must

    try:
        r = es.es.search(
            index=idx,
            body={
                "from": from_,
                "size": page_size,
                "_source": ["content_with_weight", "docnm_kwd", "doc_type_kwd",
                            "knowledge_graph_kwd"],
                "query": query,
                "sort": [{"_doc": "asc"}],
            },
        )
        total = r["hits"]["total"]["value"]
        chunks = []
        for h in r["hits"]["hits"]:
            src = h["_source"]
            content = src.get("content_with_weight", "")
            chunks.append({
                "chunk_id": h["_id"],
                "content_preview": content[:200],
                "content_full": content,
                "docnm_kwd": src.get("docnm_kwd", ""),
                "doc_type_kwd": src.get("doc_type_kwd", "text"),
                "char_count": len(content),
            })
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "chunks": chunks,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to list chunks: {e}")


# ==================== 文档管理 ====================


@app.delete("/api/document/{kb_id}/{doc_name:path}")
async def delete_document(kb_id: str, doc_name: str):
    """删除知识库中的指定文档及其所有分块"""
    es = get_es()
    idx = index_name(kb_id)
    if not es.index_exist(idx):
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")

    try:
        r = es.es.delete_by_query(
            index=idx,
            body={
                "query": {
                    "term": {"docnm_kwd": doc_name}
                }
            },
            refresh=True,
        )
        deleted = r.get("deleted", 0)
        logger.info(f"Deleted {deleted} chunks for doc '{doc_name}' from kb '{kb_id}'")
        return {
            "status": "ok",
            "deleted_chunks": deleted,
            "doc_name": doc_name,
            "kb_id": kb_id,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to delete document: {e}")


@app.post("/api/document/upload")
async def upload_document(
    kb_id: str = Form(...),
    lang: str = Form("Chinese"),
    file: UploadFile = File(...),
):
    """上传文档并解析入库"""
    es = get_es()
    idx = index_name(kb_id)

    if not es.index_exist(idx):
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found. Create it first.")

    # 读取文件
    binary = await file.read()
    filename = file.filename or "unknown.txt"
    doc_id = hashlib.md5((filename + str(time.time())).encode()).hexdigest()[:16]

    logger.info(f"Uploading {filename} to kb={kb_id}")

    # 分块
    chunks = chunk(filename, binary=binary, lang=lang)
    if not chunks:
        return {"status": "empty", "message": "No content extracted from document"}

    for ck in chunks:
        ck["doc_id"] = doc_id
        ck["kb_id"] = kb_id
        ck["knowledge_graph_kwd"] = "chunk"

    # Embedding (只对非 parent 块做 embedding，parent 块不参与检索)
    emb_mdl = get_emb()
    emb_chunks = [ck for ck in chunks if ck.get("chunk_type_kwd") != "parent"]
    texts = [ck.get("content_with_weight", "") or " " for ck in emb_chunks]

    batch_size = 16
    for i in range(0, len(emb_chunks), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_chunks = emb_chunks[i:i + batch_size]
        try:
            embeddings, _ = emb_mdl.encode(batch_texts)
            for ck, emb in zip(batch_chunks, embeddings):
                dim = len(emb)
                ck[f"q_{dim}_vec"] = emb.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise HTTPException(500, f"Embedding failed: {e}")

    # 写入 ES
    errors = es.insert(chunks, idx)
    try:
        es.es.indices.refresh(index=idx)
    except Exception:
        pass

    return {
        "status": "ok",
        "doc_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "errors": errors[:5] if errors else [],
    }


# ==================== 检索 ====================

@app.post("/api/retrieval", response_model=RetrievalResponse)
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


@app.post("/api/graph_retrieval", response_model=GraphRetrievalResponse)
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
    # 这两步互不依赖，并行可省 ~2-3s（LLM 改写时间）
    import asyncio

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

    # GraphRAG 查询改写（先启动，利用等待 ES 的时间完成 LLM 调用）
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

    # ===== Step 3: GraphRAG 图谱检索（使用预计算的 QueryAnalysis）=====
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

    # ===== Step 5: 组装最终 chunks（图谱上下文在最前面）=====
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


# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=9380, reload=True)

