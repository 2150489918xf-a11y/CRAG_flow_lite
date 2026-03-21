"""
文档管理路由 (Document Upload / List / Delete)
"""
import hashlib
import logging
import time
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form

from api.deps import get_es, get_emb
from api.errors import NotFoundError, ExternalServiceError, ok_response
from rag.app.chunking import chunk
from rag.nlp.search import index_name

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["文档管理"])


@router.get("/documents/{kb_id}")
async def list_documents(kb_id: str):
    """列出知识库中的所有文档及其分块数"""
    es = get_es()
    idx = index_name(kb_id)
    if not es.index_exist(idx):
        raise NotFoundError("知识库", kb_id)

    try:
        r = es.search_raw(
            index_name=idx,
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
        return ok_response({"documents": docs, "total": len(docs)})
    except Exception as e:
        raise ExternalServiceError("Elasticsearch", f"列出文档失败: {e}")


@router.get("/chunks/{kb_id}")
async def list_chunks(kb_id: str, page: int = 1, page_size: int = 20,
                      doc_names: Optional[str] = None):
    """查看知识库的分块内容（分页，支持按文档筛选）"""
    es = get_es()
    idx = index_name(kb_id)
    if not es.index_exist(idx):
        raise NotFoundError("知识库", kb_id)

    from_ = (page - 1) * page_size

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
        r = es.search_raw(
            index_name=idx,
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
        return ok_response({
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "chunks": chunks,
        })
    except Exception as e:
        raise ExternalServiceError("Elasticsearch", f"列出分块失败: {e}")


@router.delete("/document/{kb_id}/{doc_name:path}")
async def delete_document(kb_id: str, doc_name: str):
    """删除知识库中的指定文档及其所有分块"""
    es = get_es()
    idx = index_name(kb_id)
    if not es.index_exist(idx):
        raise NotFoundError("知识库", kb_id)

    try:
        deleted = es.delete_by_query(
            index_name=idx,
            body={
                "query": {
                    "term": {"docnm_kwd": doc_name}
                }
            },
        )
        logger.info(f"Deleted {deleted} chunks for doc '{doc_name}' from kb '{kb_id}'")
        return ok_response({
            "deleted_chunks": deleted,
            "doc_name": doc_name,
            "kb_id": kb_id,
        })
    except Exception as e:
        raise ExternalServiceError("Elasticsearch", f"删除文档失败: {e}")


@router.post("/document/upload")
async def upload_document(
    kb_id: str = Form(...),
    lang: str = Form("Chinese"),
    file: UploadFile = File(...),
):
    """上传文档并解析入库"""
    es = get_es()
    idx = index_name(kb_id)

    if not es.index_exist(idx):
        raise NotFoundError("知识库", kb_id)

    binary = await file.read()
    filename = file.filename or "unknown.txt"
    doc_id = hashlib.md5((filename + str(time.time())).encode()).hexdigest()[:16]

    logger.info(f"Uploading {filename} to kb={kb_id}")

    # 分块
    chunks = chunk(filename, binary=binary, lang=lang)
    if not chunks:
        return ok_response({"filename": filename, "chunks": 0}, message="empty")

    for ck in chunks:
        ck["doc_id"] = doc_id
        ck["kb_id"] = kb_id
        ck["knowledge_graph_kwd"] = "chunk"

    # Embedding (只对非 parent 块做 embedding)
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
            raise ExternalServiceError("Embedding", str(e))

    # 写入
    errors = es.insert(chunks, idx)
    try:
        es.refresh_index(idx)
    except Exception:
        pass

    return ok_response({
        "doc_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "errors": errors[:5] if errors else [],
    })
