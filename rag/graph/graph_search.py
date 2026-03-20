"""
GraphRAG 多路检索引擎

在线阶段：
1. 查询改写 — LLM 提取目标实体类型 + 关键实体
2. 四路并行检索 — 实体向量/实体类型/关系向量/N跳扩展
3. PageRank 加权融合排序
4. 格式化输出 — CSV 表格注入 Prompt
"""
import asyncio
import logging
import time
from typing import Optional
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np

from rag.llm.chat import ChatClient
from rag.llm.embedding import RemoteEmbedding
from rag.utils.doc_store_conn import DocStoreConnection, get_doc_store
from rag.graph.graph_store import GraphStore
from rag.nlp.search import index_name

logger = logging.getLogger(__name__)

# ==================== 查询改写 Prompt ====================

QUERY_REWRITE_SYSTEM = """你是一个查询分析助手。分析用户问题，提取关键信息用于知识图谱检索。

输出严格使用以下 JSON 格式（不要添加任何其他内容）：
```json
{
  "target_types": ["目标实体类型1", "目标实体类型2"],
  "entities": ["实体1", "实体2"],
  "keywords": ["关键词1", "关键词2"],
  "intent": "简短描述用户意图"
}
```

实体类型使用：PERSON, ORGANIZATION, LOCATION, EVENT, PRODUCT, CONCEPT, DATE, OTHER

示例：
问题："向 ChatGPT 背后公司投资的那个公司的 CEO 是谁？"
```json
{
  "target_types": ["PERSON"],
  "entities": ["ChatGPT", "投资", "CEO"],
  "keywords": ["ChatGPT", "投资", "CEO", "公司"],
  "intent": "查找投资了ChatGPT背后公司的公司的CEO"
}
```"""

QUERY_REWRITE_USER = """分析以下问题：

{question}"""


@dataclass
class QueryAnalysis:
    target_types: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    keywords: list = field(default_factory=list)
    intent: str = ""


@dataclass
class GraphSearchResult:
    """图谱检索结果"""
    entities: list = field(default_factory=list)   # 命中的实体
    relations: list = field(default_factory=list)   # 命中的关系
    paths: list = field(default_factory=list)        # N跳路径
    formatted_context: str = ""                      # 格式化的上下文文本


class GraphSearcher:
    """
    GraphRAG 多路检索引擎

    执行流程：查询改写 → 四路并行检索 → PageRank 融合 → 格式化输出
    """

    # 查询改写缓存：最多 128 条，5 分钟 TTL
    _CACHE_MAX = 128
    _CACHE_TTL = 300  # 秒

    def __init__(self, es_conn: DocStoreConnection = None, emb_mdl: RemoteEmbedding = None,
                 chat_client: ChatClient = None, graph_store: GraphStore = None):
        self.es_conn = es_conn or get_doc_store()
        self.emb_mdl = emb_mdl
        self.chat = chat_client or ChatClient()
        self.graph_store = graph_store
        self._rewrite_cache: OrderedDict[str, tuple[float, 'QueryAnalysis']] = OrderedDict()

    # ==================== Step 1: 查询改写 ====================

    async def rewrite_query(self, question: str) -> QueryAnalysis:
        """LLM 查询改写 — 提取目标实体类型和关键实体（含 LRU 缓存）"""
        # 缓存命中检查
        cache_key = question.strip().lower()
        if cache_key in self._rewrite_cache:
            ts, cached = self._rewrite_cache[cache_key]
            if time.time() - ts < self._CACHE_TTL:
                logger.info(f"Query rewrite cache HIT: {question[:30]}...")
                self._rewrite_cache.move_to_end(cache_key)
                return cached
            else:
                del self._rewrite_cache[cache_key]

        # 缓存未命中 → 调 LLM
        try:
            result = await self.chat.achat_json(
                QUERY_REWRITE_SYSTEM,
                QUERY_REWRITE_USER.format(question=question),
                temperature=0.1,
            )
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return QueryAnalysis(
                entities=[question],
                keywords=question.split(),
                intent=question,
            )

        qa = QueryAnalysis(
            target_types=[t.upper() for t in result.get("target_types", [])],
            entities=result.get("entities", []),
            keywords=result.get("keywords", question.split()),
            intent=result.get("intent", question),
        )

        # 写入缓存
        self._rewrite_cache[cache_key] = (time.time(), qa)
        if len(self._rewrite_cache) > self._CACHE_MAX:
            self._rewrite_cache.popitem(last=False)

        return qa

    # ==================== Step 2: 四路并行检索 ====================

    async def _search_entity_by_vector(self, idx_names, query_text, topk=20):
        """路线1：实体向量检索"""
        if not self.emb_mdl:
            return []
        qv, _ = self.emb_mdl.encode_queries(query_text)
        dim = len(qv)

        from rag.nlp.query import MatchDenseExpr, FusionExpr

        bool_filter = {
            "bool": {
                "filter": [{"term": {"knowledge_graph_kwd": "entity"}}]
            }
        }

        body = {
            "knn": {
                "field": f"q_{dim}_vec",
                "query_vector": qv.tolist(),
                "k": topk,
                "num_candidates": topk * 2,
                "filter": {"term": {"knowledge_graph_kwd": "entity"}},
            },
            "size": topk,
            "_source": True,
        }

        try:
            res = self.es_conn.es.search(index=idx_names, body=body, timeout="30s")
            return [
                {**h["_source"], "id": h["_id"], "_score": h.get("_score", 0)}
                for h in res.get("hits", {}).get("hits", [])
            ]
        except Exception as e:
            logger.warning(f"Entity vector search failed: {e}")
            return []

    async def _search_entity_by_type(self, idx_names, target_types, topk=50):
        """路线2：实体类型检索"""
        if not target_types:
            return []

        body = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"knowledge_graph_kwd": "entity"}},
                        {"terms": {"entity_type_kwd": target_types}},
                    ]
                }
            },
            "sort": [{"pagerank_flt": {"order": "desc", "unmapped_type": "float"}}],
            "size": topk,
            "_source": True,
        }

        try:
            res = self.es_conn.es.search(index=idx_names, body=body, timeout="30s")
            return [
                {**h["_source"], "id": h["_id"], "_score": h.get("_score", 0)}
                for h in res.get("hits", {}).get("hits", [])
            ]
        except Exception as e:
            logger.warning(f"Entity type search failed: {e}")
            return []

    async def _search_relation_by_vector(self, idx_names, question, topk=20):
        """路线3：关系向量检索"""
        if not self.emb_mdl:
            return []
        qv, _ = self.emb_mdl.encode_queries(question)
        dim = len(qv)

        body = {
            "knn": {
                "field": f"q_{dim}_vec",
                "query_vector": qv.tolist(),
                "k": topk,
                "num_candidates": topk * 2,
                "filter": {"term": {"knowledge_graph_kwd": "relation"}},
            },
            "size": topk,
            "_source": True,
        }

        try:
            res = self.es_conn.es.search(index=idx_names, body=body, timeout="30s")
            return [
                {**h["_source"], "id": h["_id"], "_score": h.get("_score", 0)}
                for h in res.get("hits", {}).get("hits", [])
            ]
        except Exception as e:
            logger.warning(f"Relation vector search failed: {e}")
            return []

    async def _expand_nhop(self, entities, n_hops=2, topk=20):
        """路线4：N跳路径扩展"""
        if not self.graph_store or not entities:
            return []

        paths = []
        for entity_doc in entities[:5]:  # 只对前 5 个实体做扩展
            name = entity_doc.get("entity_name_kwd", "")
            if not name:
                continue
            neighbors = self.graph_store.get_neighbors(name, n_hops=n_hops)
            for nb in neighbors[:topk]:
                paths.append({
                    "from": name,
                    "to": nb["name"],
                    "type": nb["type"],
                    "relation": nb["relation"],
                    "hop": nb["hop"],
                    "pagerank": nb["pagerank"],
                    "description": nb.get("description", ""),
                })
        return paths

    # ==================== Step 3: PageRank 融合排序 ====================

    def _fusion_score(self, doc, query_score_weight=0.5):
        """
        贝叶斯融合排序：Score = 向量相似度 × PageRank

        Args:
            doc: ES 文档（包含 _score 和 pagerank_flt）
            query_score_weight: 查询相似度权重（保留但目前用乘法）
        """
        vector_sim = float(doc.get("_score", 0.0))
        pagerank = float(doc.get("pagerank_flt", 0.01))
        # 确保 pagerank 不为 0
        pagerank = max(pagerank, 0.01)
        return vector_sim * pagerank

    def _rank_results(self, entity_results, relation_results, paths):
        """对所有检索结果做 PageRank 融合排序"""

        # 实体去重并打分
        entity_map = {}
        for doc in entity_results:
            name = doc.get("entity_name_kwd", "")
            if not name:
                continue
            key = name.lower()
            score = self._fusion_score(doc)
            if key not in entity_map or score > entity_map[key]["fusion_score"]:
                entity_map[key] = {
                    "name": name,
                    "type": doc.get("entity_type_kwd", "OTHER"),
                    "description": doc.get("content_with_weight", ""),
                    "pagerank": float(doc.get("pagerank_flt", 0)),
                    "fusion_score": score,
                }

        # 关系打分
        ranked_relations = []
        seen_rels = set()
        for doc in relation_results:
            src = doc.get("entity_name_kwd", "")
            tgt = doc.get("target_entity_kwd", "")
            rel_key = f"{src}|{tgt}"
            if rel_key in seen_rels:
                continue
            seen_rels.add(rel_key)
            score = self._fusion_score(doc)
            ranked_relations.append({
                "source": src,
                "target": tgt,
                "description": doc.get("content_with_weight", ""),
                "pagerank": float(doc.get("pagerank_flt", 0)),
                "fusion_score": score,
            })

        # 排序
        ranked_entities = sorted(entity_map.values(), key=lambda x: x["fusion_score"], reverse=True)
        ranked_relations.sort(key=lambda x: x["fusion_score"], reverse=True)

        return ranked_entities, ranked_relations

    # ==================== Step 4: 格式化输出 ====================

    @staticmethod
    def format_context(entities, relations, paths=None, max_entities=10, max_relations=15):
        """
        将图谱结果格式化为 CSV 表格字符串

        Returns:
            str: 可直接注入到 Prompt 中的上下文文本（无内容时返回空字符串）
        """
        # 如果没有任何图谱数据，直接返回空
        if not entities and not relations and not paths:
            return ""

        lines = []
        lines.append("=== 知识图谱上下文 ===")
        lines.append("")

        # 实体表格
        if entities:
            lines.append("【实体信息】")
            lines.append("实体名称 | 实体类型 | 描述 | 重要度")
            lines.append("--- | --- | --- | ---")
            for e in entities[:max_entities]:
                lines.append(
                    f"{e['name']} | {e['type']} | {e['description']} | {e['pagerank']:.3f}"
                )
            lines.append("")

        # 关系表格
        if relations:
            lines.append("【关系信息】")
            lines.append("源实体 | 关系 | 目标实体 | 重要度")
            lines.append("--- | --- | --- | ---")
            for r in relations[:max_relations]:
                lines.append(
                    f"{r['source']} | {r['description']} | {r['target']} | {r['pagerank']:.3f}"
                )
            lines.append("")

        # N跳路径
        if paths:
            lines.append("【推理路径】")
            for p in paths[:10]:
                lines.append(
                    f"  {p['from']} --[{p['relation']}]--> {p['to']} (类型: {p['type']}, 跳数: {p['hop']})"
                )
            lines.append("")

        lines.append("=== 知识图谱上下文结束 ===")
        return "\n".join(lines)

    # ==================== 主检索入口 ====================

    async def search(self, question: str, kb_ids: list,
                     topk_entity=20, topk_relation=20, n_hops=2) -> GraphSearchResult:
        """
        GraphRAG 完整检索流程

        Args:
            question: 用户问题
            kb_ids: 知识库 ID 列表
            topk_entity: 实体检索数
            topk_relation: 关系检索数
            n_hops: N跳扩展跳数

        Returns:
            GraphSearchResult 包含排序后的实体/关系/路径和格式化上下文
        """
        # Step 1: 查询改写
        logger.info(f"GraphRAG: analyzing query: {question[:50]}...")
        qa = await self.rewrite_query(question)
        logger.info(f"  Query analysis: types={qa.target_types}, entities={qa.entities}")

        return await self.search_with_qa(question, kb_ids, qa,
                                          topk_entity, topk_relation, n_hops)

    async def search_with_qa(self, question: str, kb_ids: list,
                              qa: QueryAnalysis,
                              topk_entity=20, topk_relation=20,
                              n_hops=2) -> GraphSearchResult:
        """
        GraphRAG 检索（使用预计算的 QueryAnalysis，适用于并行优化场景）
        """
        idx_names = [index_name(kb_id) for kb_id in kb_ids]

        # Step 2: 四路并行检索
        entity_query = " ".join(qa.entities) if qa.entities else question

        entity_vec_results, entity_type_results, relation_results = await asyncio.gather(
            self._search_entity_by_vector(idx_names, entity_query, topk_entity),
            self._search_entity_by_type(idx_names, qa.target_types, topk_entity),
            self._search_relation_by_vector(idx_names, question, topk_relation),
        )

        # 合并实体检索结果
        all_entity_results = entity_vec_results + entity_type_results

        # Step 2.5: N跳扩展
        paths = await self._expand_nhop(entity_vec_results, n_hops=n_hops)

        # Step 3: PageRank 融合排序
        ranked_entities, ranked_relations = self._rank_results(all_entity_results, relation_results, paths)

        logger.info(f"  GraphRAG results: {len(ranked_entities)} entities, "
                     f"{len(ranked_relations)} relations, {len(paths)} paths")

        # Step 4: 格式化输出
        context = self.format_context(ranked_entities, ranked_relations, paths)

        return GraphSearchResult(
            entities=ranked_entities,
            relations=ranked_relations,
            paths=paths,
            formatted_context=context,
        )

    async def enhanced_retrieval(self, question: str, kb_ids: list,
                                 text_chunks: list = None,
                                 max_entities=10, max_relations=15) -> list:
        """
        增强检索：将图谱上下文注入到文本 Chunk 前面

        Args:
            question: 用户问题
            kb_ids: 知识库 ID 列表
            text_chunks: 普通文本 chunk 列表
            max_entities: 最大实体数
            max_relations: 最大关系数

        Returns:
            list: [graph_context, chunk1, chunk2, ...] 图谱上下文在最前面
        """
        graph_result = await self.search(question, kb_ids)

        result_chunks = list(text_chunks) if text_chunks else []

        # 将图谱上下文强制插入最前面
        if graph_result.formatted_context.strip():
            graph_chunk = {
                "chunk_id": "graph_context",
                "content_with_weight": graph_result.formatted_context,
                "docnm_kwd": "[知识图谱]",
                "doc_type_kwd": "knowledge_graph",
                "similarity": 1.0,
                "is_graph_context": True,
            }
            result_chunks.insert(0, graph_chunk)

        return result_chunks
