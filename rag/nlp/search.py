"""
混合检索引擎 (精简自 RAGFlow rag/nlp/search.py)
实现 fulltext + KNN 混合检索和重排序
"""
import logging
import math
import re
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from rag.nlp.query import FulltextQueryer, MatchTextExpr, MatchDenseExpr, FusionExpr
from rag.nlp import tokenizer as rag_tokenizer_module
from rag.utils.es_conn import ESConnection

rag_tokenizer = rag_tokenizer_module
logger = logging.getLogger(__name__)


def index_name(kb_id):
    """根据知识库 ID 生成 ES 索引名"""
    return f"ragflow_lite_{kb_id}"


class Dealer:
    """
    检索引擎核心
    照搬 RAGFlow search.py Dealer 的检索、重排逻辑
    """

    def __init__(self, es_conn: ESConnection = None):
        self.qryr = FulltextQueryer()
        self.es_conn = es_conn or ESConnection()

    @dataclass
    class SearchResult:
        total: int
        ids: list
        query_vector: list = None
        field: dict = None
        highlight: dict = None
        keywords: list = None

    async def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        """获取查询向量"""
        qv, _ = emb_mdl.encode_queries(txt)
        embedding_data = [float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk,
                              {"similarity": similarity})

    async def search(self, req, idx_names, emb_mdl=None, highlight=False):
        """
        混合检索
        """
        if highlight is None:
            highlight = False

        condition = {}
        for key, field in {"doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]

        topk = int(req.get("topk", 1024))
        offset = 0
        limit = topk

        src = [
            "docnm_kwd", "content_ltks", "kb_id", "title_tks",
            "important_kwd", "doc_id", "content_with_weight",
            "question_kwd", "question_tks", "doc_type_kwd",
        ]
        kwds = set()

        qst = req.get("question", "")
        q_vec = []

        if not qst:
            res = self.es_conn.search(src, [], condition, [], offset, limit, idx_names)
            total = self.es_conn.get_total(res)
        else:
            highlight_fields = ["content_ltks", "title_tks"] if highlight else []
            matchText, keywords = self.qryr.question(qst, min_match=0.3)

            if emb_mdl is None:
                matchExprs = [matchText] if matchText else []
                res = self.es_conn.search(src, highlight_fields, condition,
                                          matchExprs, offset, limit, idx_names)
                total = self.es_conn.get_total(res)
            else:
                matchDense = await self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
                q_vec = matchDense.embedding_data
                src.append(f"q_{len(q_vec)}_vec")

                # RAGFlow 的融合权重: 0.05 fulltext + 0.95 vector
                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
                matchExprs = []
                if matchText:
                    matchExprs.append(matchText)
                matchExprs.extend([matchDense, fusionExpr])

                res = self.es_conn.search(src, highlight_fields, condition,
                                          matchExprs, offset, limit, idx_names)
                total = self.es_conn.get_total(res)

                # 结果为空时降低匹配阈值重试
                if total == 0 and matchText:
                    matchText, _ = self.qryr.question(qst, min_match=0.1)
                    matchDense.extra_options["similarity"] = 0.17
                    matchExprs = [matchText, matchDense, fusionExpr]
                    res = self.es_conn.search(src, highlight_fields, condition,
                                              matchExprs, offset, limit, idx_names)
                    total = self.es_conn.get_total(res)

            if keywords:
                for k in keywords:
                    kwds.add(k)
                    for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                        if len(kk) < 2:
                            continue
                        kwds.add(kk)

        ids = self.es_conn.get_doc_ids(res)
        keywords_list = list(kwds)
        hl = self.es_conn.get_highlight(res, keywords_list, "content_with_weight")
        fields = self.es_conn.get_fields(res, src + ["_score"])

        return self.SearchResult(
            total=total,
            ids=ids,
            query_vector=q_vec,
            highlight=hl,
            field=fields,
            keywords=keywords_list,
        )

    def rerank(self, sres, query, tkweight=0.3, vtweight=0.7):
        """
        重排序
        照搬 RAGFlow 的 hybrid_similarity 融合公式
        """
        _, keywords = self.qryr.question(query)
        if not sres.query_vector:
            return [], [], []

        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size

        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field.get(chunk_id, {}).get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [float(v) for v in vector.split("\t")]
            ins_embd.append(vector)

        if not ins_embd:
            return [], [], []

        ins_tw = []
        for i in sres.ids:
            content_ltks = list(OrderedDict.fromkeys(
                sres.field.get(i, {}).get("content_ltks", "").split()
            ))
            title_tks = [t for t in sres.field.get(i, {}).get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field.get(i, {}).get("question_tks", "").split() if t]
            important_kwd = sres.field.get(i, {}).get("important_kwd", [])
            if isinstance(important_kwd, str):
                important_kwd = [important_kwd]
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(
            sres.query_vector, ins_embd, keywords, ins_tw, tkweight, vtweight
        )
        return sim, tksim, vtsim

    async def retrieval(self, question, embd_mdl, kb_ids,
                        page=1, page_size=5,
                        similarity_threshold=0.2,
                        vector_similarity_weight=0.3,
                        top=1024, doc_ids=None,
                        highlight=False):
        """
        完整检索流程：search → rerank → 分页 → 返回
        """
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks

        idx_names = [index_name(kb_id) for kb_id in kb_ids]

        req = {
            "doc_ids": doc_ids,
            "page": 1,
            "size": top,
            "question": question,
            "topk": top,
            "similarity": similarity_threshold,
        }

        sres = await self.search(req, idx_names, embd_mdl, highlight)

        if sres.total == 0 or not sres.query_vector:
            ranks["total"] = sres.total
            return ranks

        # 重排序
        sim, tsim, vsim = self.rerank(
            sres, question,
            1 - vector_similarity_weight,
            vector_similarity_weight,
        )

        if not hasattr(sim, '__len__') or len(sim) == 0:
            return ranks

        sim_np = np.array(sim, dtype=np.float64)
        sorted_idx = np.argsort(sim_np * -1)

        post_threshold = 0.0 if vector_similarity_weight <= 0 else similarity_threshold
        valid_idx = [int(i) for i in sorted_idx if sim_np[i] >= post_threshold]
        ranks["total"] = len(valid_idx)

        if not valid_idx:
            ranks["doc_aggs"] = []
            return ranks

        # 分页
        begin = (page - 1) * page_size
        end = begin + page_size
        page_idx = valid_idx[begin:end]

        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim

        for i in page_idx:
            id = sres.ids[i]
            chunk = sres.field.get(id, {})
            d = {
                "chunk_id": id,
                "content_ltks": chunk.get("content_ltks", ""),
                "content_with_weight": chunk.get("content_with_weight", ""),
                "doc_id": chunk.get("doc_id", ""),
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "kb_id": chunk.get("kb_id", ""),
                "important_kwd": chunk.get("important_kwd", []),
                "similarity": float(sim_np[i]),
                "vector_similarity": float(vsim[i]) if vsim is not None and len(vsim) > i else 0.0,
                "term_similarity": float(tsim[i]) if tsim is not None and len(tsim) > i else 0.0,
                "vector": chunk.get(vector_column, zero_vector),
            }
            if highlight and sres.highlight:
                d["highlight"] = sres.highlight.get(id, d["content_with_weight"])
            ranks["chunks"].append(d)

        # 文档聚合
        doc_aggs = {}
        for i in valid_idx:
            id = sres.ids[i]
            chunk = sres.field.get(id, {})
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")
            if dnm not in doc_aggs:
                doc_aggs[dnm] = {"doc_id": did, "count": 0}
            doc_aggs[dnm]["count"] += 1

        ranks["doc_aggs"] = [
            {"doc_name": k, "doc_id": v["doc_id"], "count": v["count"]}
            for k, v in sorted(doc_aggs.items(), key=lambda x: x[1]["count"] * -1)
        ]

        return ranks
