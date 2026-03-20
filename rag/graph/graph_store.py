"""
图谱存储：NetworkX 内存图 + PageRank + ES 扁平化存储

职责：
1. 从 ExtractionResult 构建 NetworkX 有向图
2. 计算 PageRank 得分
3. 将实体/关系"拍平"存入 ES（与 Chunk 同索引，用 knowledge_graph_kwd 区分）
"""
import hashlib
import logging
from typing import Optional

import networkx as nx
import numpy as np

from rag.graph.extractor import Entity, Relation, ExtractionResult
from rag.llm.embedding import RemoteEmbedding
from rag.utils.doc_store_conn import DocStoreConnection, get_doc_store
from rag.nlp.tokenizer import tokenizer as rag_tokenizer

logger = logging.getLogger(__name__)


class GraphStore:
    """
    图谱存储管理器

    - 内存图：NetworkX DiGraph（用于 PageRank 和 N 跳遍历）
    - 持久化：ES 索引（实体和关系作为扁平文档存储）
    """

    def __init__(self, es_conn: DocStoreConnection = None, emb_mdl: RemoteEmbedding = None):
        self.es_conn = es_conn or get_doc_store()
        self.emb_mdl = emb_mdl
        self.graph = nx.DiGraph()  # 内存图
        self._entity_map = {}      # name -> Entity

    def build_graph(self, extraction: ExtractionResult):
        """
        从提取结果构建 NetworkX 有向图

        Args:
            extraction: LLM 提取的实体和关系
        """
        # 去重实体（按名称合并）
        entity_map = {}
        for e in extraction.entities:
            key = e.name.lower().strip()
            if key not in entity_map:
                entity_map[key] = e
            elif len(e.description) > len(entity_map[key].description):
                entity_map[key].description = e.description

        # 添加节点
        for key, entity in entity_map.items():
            self.graph.add_node(key, **{
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
                "chunk_id": entity.chunk_id,
            })

        # 添加边
        for rel in extraction.relations:
            src = rel.source.lower().strip()
            tgt = rel.target.lower().strip()
            # 确保两端节点都存在
            if src not in self.graph:
                self.graph.add_node(src, name=rel.source, type="OTHER",
                                    description="", chunk_id=rel.chunk_id)
            if tgt not in self.graph:
                self.graph.add_node(tgt, name=rel.target, type="OTHER",
                                    description="", chunk_id=rel.chunk_id)
            self.graph.add_edge(src, tgt, description=rel.description,
                                chunk_id=rel.chunk_id)

        self._entity_map = {k: v for k, v in entity_map.items()}
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, "
                     f"{self.graph.number_of_edges()} edges")

    def compute_pagerank(self, alpha=0.85):
        """
        计算 PageRank 得分

        Args:
            alpha: 阻尼系数 (默认 0.85)

        Returns:
            dict: {node_key: pagerank_score}
        """
        if self.graph.number_of_nodes() == 0:
            return {}

        try:
            pr = nx.pagerank(self.graph, alpha=alpha)
        except Exception as e:
            logger.warning(f"PageRank computation failed: {e}, using uniform scores")
            n = self.graph.number_of_nodes()
            pr = {node: 1.0 / n for node in self.graph.nodes()}

        # 归一化到 [0, 1]
        max_pr = max(pr.values()) if pr else 1.0
        if max_pr > 0:
            pr = {k: v / max_pr for k, v in pr.items()}

        # 更新节点属性
        for node, score in pr.items():
            self.graph.nodes[node]["pagerank"] = score
            if node in self._entity_map:
                self._entity_map[node].pagerank = score

        logger.info(f"PageRank computed for {len(pr)} nodes, "
                     f"max={max(pr.values()):.4f}, min={min(pr.values()):.4f}")
        return pr

    def get_neighbors(self, entity_name, n_hops=2):
        """
        获取 N 跳邻居实体

        Args:
            entity_name: 起始实体名
            n_hops: 跳数 (1-2)

        Returns:
            list[dict]: 邻居实体及路径信息
        """
        key = entity_name.lower().strip()
        if key not in self.graph:
            return []

        neighbors = []
        visited = {key}

        # BFS N 跳
        frontier = {key}
        for hop in range(1, n_hops + 1):
            next_frontier = set()
            for node in frontier:
                # 出边
                for _, tgt, data in self.graph.out_edges(node, data=True):
                    if tgt not in visited:
                        visited.add(tgt)
                        next_frontier.add(tgt)
                        node_data = self.graph.nodes[tgt]
                        neighbors.append({
                            "name": node_data.get("name", tgt),
                            "type": node_data.get("type", "OTHER"),
                            "description": node_data.get("description", ""),
                            "pagerank": node_data.get("pagerank", 0.0),
                            "hop": hop,
                            "relation": data.get("description", ""),
                            "from": self.graph.nodes[node].get("name", node),
                        })
                # 入边
                for src, _, data in self.graph.in_edges(node, data=True):
                    if src not in visited:
                        visited.add(src)
                        next_frontier.add(src)
                        node_data = self.graph.nodes[src]
                        neighbors.append({
                            "name": node_data.get("name", src),
                            "type": node_data.get("type", "OTHER"),
                            "description": node_data.get("description", ""),
                            "pagerank": node_data.get("pagerank", 0.0),
                            "hop": hop,
                            "relation": data.get("description", ""),
                            "from": self.graph.nodes[node].get("name", node),
                        })
            frontier = next_frontier

        # 按 PageRank 排序
        neighbors.sort(key=lambda x: x["pagerank"], reverse=True)
        return neighbors

    async def store_to_es(self, index_name: str, kb_id: str,
                          extraction: ExtractionResult, pagerank_scores: dict):
        """
        将图谱数据"拍平"存入 ES
        - 实体记录: knowledge_graph_kwd = "entity"
        - 关系记录: knowledge_graph_kwd = "relation"

        Args:
            index_name: ES 索引名
            kb_id: 知识库 ID
            extraction: 实体和关系
            pagerank_scores: PageRank 得分
        """
        if not self.emb_mdl:
            raise ValueError("Embedding model required for graph storage")

        documents = []

        # ---- 实体文档 ----
        entity_texts = []
        entity_items = []
        seen_entities = set()

        for entity in extraction.entities:
            key = entity.name.lower().strip()
            if key in seen_entities:
                continue
            seen_entities.add(key)

            desc = entity.description or entity.name
            entity_texts.append(desc)
            entity_items.append(entity)

        if entity_texts:
            embeddings, _ = self.emb_mdl.encode(entity_texts)
            dim = len(embeddings[0])

            for entity, emb in zip(entity_items, embeddings):
                key = entity.name.lower().strip()
                pr_score = pagerank_scores.get(key, 0.0)
                doc_id = "entity_" + hashlib.md5(entity.name.encode()).hexdigest()[:16]

                doc = {
                    "id": doc_id,
                    "kb_id": kb_id,
                    "knowledge_graph_kwd": "entity",
                    "entity_name_kwd": entity.name,
                    "entity_type_kwd": entity.type,
                    "content_with_weight": entity.description or entity.name,
                    "content_ltks": rag_tokenizer.tokenize(entity.description or entity.name),
                    "content_sm_ltks": rag_tokenizer.fine_grained_tokenize(
                        rag_tokenizer.tokenize(entity.description or entity.name)
                    ),
                    "pagerank_flt": float(pr_score),
                    f"q_{dim}_vec": emb.tolist(),
                    "docnm_kwd": f"[KG] {entity.name}",
                    "doc_id": entity.chunk_id or doc_id,
                    "doc_type_kwd": "knowledge_graph",
                }
                documents.append(doc)

        # ---- 关系文档 ----
        relation_texts = []
        relation_items = []
        seen_relations = set()

        for rel in extraction.relations:
            rel_key = f"{rel.source}|{rel.target}|{rel.description}"
            if rel_key in seen_relations:
                continue
            seen_relations.add(rel_key)

            desc = f"{rel.source} {rel.description} {rel.target}"
            relation_texts.append(desc)
            relation_items.append(rel)

        if relation_texts:
            embeddings, _ = self.emb_mdl.encode(relation_texts)
            dim = len(embeddings[0])

            for rel, emb in zip(relation_items, embeddings):
                doc_id = "relation_" + hashlib.md5(
                    f"{rel.source}_{rel.target}".encode()
                ).hexdigest()[:16]

                # 关系两端实体的 PageRank 取较大值
                src_pr = pagerank_scores.get(rel.source.lower().strip(), 0.0)
                tgt_pr = pagerank_scores.get(rel.target.lower().strip(), 0.0)
                rel_pr = max(src_pr, tgt_pr)

                rel_desc = f"{rel.source} {rel.description} {rel.target}"
                doc = {
                    "id": doc_id,
                    "kb_id": kb_id,
                    "knowledge_graph_kwd": "relation",
                    "entity_name_kwd": rel.source,
                    "target_entity_kwd": rel.target,
                    "content_with_weight": rel_desc,
                    "content_ltks": rag_tokenizer.tokenize(rel_desc),
                    "content_sm_ltks": rag_tokenizer.fine_grained_tokenize(
                        rag_tokenizer.tokenize(rel_desc)
                    ),
                    "pagerank_flt": float(rel_pr),
                    f"q_{dim}_vec": emb.tolist(),
                    "docnm_kwd": f"[KG] {rel.source} → {rel.target}",
                    "doc_id": rel.chunk_id or doc_id,
                    "doc_type_kwd": "knowledge_graph",
                }
                documents.append(doc)

        if documents:
            errors = self.es_conn.insert(documents, index_name)
            if errors:
                logger.warning(f"ES insert graph errors: {errors[:3]}")
            logger.info(f"Stored {len(documents)} graph documents to ES "
                         f"({len(entity_items)} entities, {len(relation_items)} relations)")

        return len(documents)

    def save_graph(self, filepath):
        """持久化 NetworkX 图到文件"""
        import json
        data = nx.node_link_data(self.graph)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Graph saved to {filepath}")

    def load_graph(self, filepath):
        """从文件加载 NetworkX 图"""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, directed=True)
        # 重建 entity map
        self._entity_map = {}
        for node, attrs in self.graph.nodes(data=True):
            self._entity_map[node] = Entity(
                name=attrs.get("name", node),
                type=attrs.get("type", "OTHER"),
                description=attrs.get("description", ""),
                pagerank=attrs.get("pagerank", 0.0),
            )
        logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, "
                     f"{self.graph.number_of_edges()} edges")
