"""
Elasticsearch 连接器 (精简自 RAGFlow rag/utils/es_conn.py)
"""
import copy
import json
import logging
import os
import re
import time

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search, UpdateByQuery

from rag.nlp.query import MatchTextExpr, MatchDenseExpr, FusionExpr
from rag.settings import get_es_config, get_project_base_directory

logger = logging.getLogger(__name__)

ATTEMPT_TIME = 2


class ESConnection:
    """
    Elasticsearch 连接和 CRUD 操作
    """

    def __init__(self, hosts=None, username=None, password=None):
        if hosts is None:
            es_cfg = get_es_config()
            hosts = es_cfg.get("hosts", "http://localhost:9200")
            username = es_cfg.get("username", "")
            password = es_cfg.get("password", "")

        kwargs = {"hosts": hosts, "timeout": 600, "retry_on_timeout": True}
        if username and password:
            kwargs["basic_auth"] = (username, password)

        self.es = Elasticsearch(**kwargs)
        logger.info(f"Connected to ES: {hosts}")

    def health(self):
        """检查 ES 健康状态"""
        return self.es.cluster.health()

    def create_idx(self, index_name, mapping_path=None, display_name=None):
        """创建索引，可选存储显示名称到 _meta"""
        if self.es.indices.exists(index=index_name):
            return True

        if mapping_path is None:
            mapping_path = os.path.join(get_project_base_directory(), "conf", "mapping.json")

        with open(mapping_path, "r") as f:
            mapping = json.load(f)

        # Store display name in _meta for Chinese name support
        if display_name:
            if "mappings" not in mapping:
                mapping["mappings"] = {}
            if "_meta" not in mapping["mappings"]:
                mapping["mappings"]["_meta"] = {}
            mapping["mappings"]["_meta"]["display_name"] = display_name

        self.es.indices.create(index=index_name, body=mapping)
        logger.info(f"Created index: {index_name} (display: {display_name or index_name})")
        return True

    def delete_idx(self, index_name):
        """删除索引"""
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
            logger.info(f"Deleted index: {index_name}")
            return True
        return False

    def index_exist(self, index_name):
        """检查索引是否存在"""
        return self.es.indices.exists(index=index_name)

    def get_index_meta(self, index_name):
        """获取索引的 _meta 信息（包含 display_name 等）"""
        try:
            mapping = self.es.indices.get_mapping(index=index_name)
            return mapping.get(index_name, {}).get("mappings", {}).get("_meta", {})
        except Exception:
            return {}

    def search(self, select_fields, highlight_fields, condition, match_expressions,
               offset, limit, index_names, rank_feature=None):
        """
        混合检索
        照搬 RAGFlow ESConnection.search 核心逻辑
        """
        if isinstance(index_names, str):
            index_names = [index_names]

        bool_query = Q("bool", must=[])

        # 构建过滤条件
        for k, v in condition.items():
            if not v:
                continue
            if isinstance(v, list):
                bool_query.filter.append(Q("terms", **{k: v}))
            elif isinstance(v, (str, int)):
                bool_query.filter.append(Q("term", **{k: v}))

        s = Search()
        vector_similarity_weight = 0.5

        # 提取融合权重
        for m in match_expressions:
            if isinstance(m, FusionExpr) and m.method == "weighted_sum" and "weights" in m.fusion_params:
                weights = m.fusion_params["weights"]
                vector_similarity_weight = float(weights.split(",")[1])

        # 构建查询
        for m in match_expressions:
            if isinstance(m, MatchTextExpr):
                minimum_should_match = m.extra_options.get("minimum_should_match", 0.0)
                if isinstance(minimum_should_match, float):
                    minimum_should_match = str(int(minimum_should_match * 100)) + "%"
                bool_query.must.append(Q("query_string",
                                         fields=m.fields,
                                         type="best_fields",
                                         query=m.matching_text,
                                         minimum_should_match=minimum_should_match,
                                         boost=1))
                bool_query.boost = 1.0 - vector_similarity_weight

            elif isinstance(m, MatchDenseExpr):
                similarity = m.extra_options.get("similarity", 0.0)
                s = s.knn(
                    m.vector_column_name,
                    m.topn,
                    m.topn * 2,
                    query_vector=list(m.embedding_data),
                    filter=bool_query.to_dict(),
                    similarity=similarity,
                )

        if bool_query:
            s = s.query(bool_query)

        for field in highlight_fields:
            s = s.highlight(field)

        if limit > 0:
            s = s[offset:offset + limit]

        q = s.to_dict()
        logger.debug(f"ES search query: {json.dumps(q, ensure_ascii=False)[:500]}")

        for i in range(ATTEMPT_TIME):
            try:
                res = self.es.search(
                    index=index_names,
                    body=q,
                    timeout="600s",
                    track_total_hits=True,
                    _source=True,
                )
                return res
            except Exception as e:
                logger.warning(f"ES search attempt {i + 1} failed: {e}")
                if i == ATTEMPT_TIME - 1:
                    raise

    def insert(self, documents, index_name):
        """批量插入文档"""
        operations = []
        for d in documents:
            d_copy = copy.deepcopy(d)
            meta_id = d_copy.get("id", "")
            operations.append({"index": {"_index": index_name, "_id": meta_id}})
            operations.append(d_copy)

        errors = []
        for _ in range(ATTEMPT_TIME):
            try:
                r = self.es.bulk(index=index_name, operations=operations,
                                 refresh=False, timeout="600s")
                if str(r.get("errors", "")).lower() == "false":
                    return errors
                for item in r.get("items", []):
                    for action in ["create", "delete", "index", "update"]:
                        if action in item and "error" in item[action]:
                            errors.append(str(item[action]["_id"]) + ":" + str(item[action]["error"]))
                return errors
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"ES insert error: {e}")
                time.sleep(1)
        return errors

    def delete(self, condition, index_name):
        """按条件删除文档"""
        bool_query = Q("bool")
        for k, v in condition.items():
            if isinstance(v, list):
                bool_query.must.append(Q("terms", **{k: v}))
            elif isinstance(v, (str, int)):
                bool_query.must.append(Q("term", **{k: v}))

        for _ in range(ATTEMPT_TIME):
            try:
                res = self.es.delete_by_query(
                    index=index_name,
                    body=Search().query(bool_query).to_dict(),
                    refresh=True
                )
                return res.get("deleted", 0)
            except Exception as e:
                logger.warning(f"ES delete error: {e}")
                time.sleep(1)
        return 0

    # ---- 辅助方法 ----

    @staticmethod
    def get_total(res):
        if not res:
            return 0
        try:
            t = res["hits"]["total"]
            return t["value"] if isinstance(t, dict) else t
        except Exception:
            return 0

    @staticmethod
    def get_doc_ids(res):
        if not res:
            return []
        return [h["_id"] for h in res.get("hits", {}).get("hits", [])]

    @staticmethod
    def get_source(res):
        if not res:
            return []
        return [
            {**h["_source"], "id": h["_id"]}
            for h in res.get("hits", {}).get("hits", [])
        ]

    @staticmethod
    def get_fields(res, fields):
        res_fields = {}
        if not res or not fields:
            return res_fields

        for h in res.get("hits", {}).get("hits", []):
            d = h.get("_source", {})
            d["id"] = h["_id"]
            m = {n: d.get(n) for n in fields if d.get(n) is not None}
            if m:
                res_fields[h["_id"]] = m
        return res_fields

    @staticmethod
    def get_highlight(res, keywords=None, field="content_with_weight"):
        highlights = {}
        if not res:
            return highlights
        for h in res.get("hits", {}).get("hits", []):
            hl = h.get("highlight", {})
            if field in hl:
                highlights[h["_id"]] = " ".join(hl[field])
        return highlights
