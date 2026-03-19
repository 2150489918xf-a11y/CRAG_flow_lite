"""
Reranker 模型客户端
支持 SiliconFlow / Jina / Cohere 等兼容接口的重排模型

使用方式：
  reranker = RemoteReranker(api_key="...", model_name="BAAI/bge-reranker-v2-m3")
  results = reranker.rerank("用户问题", ["chunk1", "chunk2", ...], top_n=5)
"""
import logging
from typing import Optional

import requests

from rag.settings import get_config

logger = logging.getLogger(__name__)


class RemoteReranker:
    """
    远程 Reranker 客户端

    支持的 API 格式 (SiliconFlow / Jina 风格):
      POST /v1/rerank
      {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "...",
        "documents": ["...", "..."],
        "top_n": 5
      }
    """

    def __init__(self, api_key=None, model_name=None, base_url=None):
        cfg = get_config().get("reranker", {})
        self.api_key = api_key or cfg.get("api_key", "")
        self.model_name = model_name or cfg.get("model_name", "BAAI/bge-reranker-v2-m3")
        self.base_url = (base_url or cfg.get("base_url", "https://api.siliconflow.cn/v1")).rstrip("/")

    def rerank(self, query: str, documents: list[str],
               top_n: int = None, return_documents: bool = False) -> list[dict]:
        """
        对候选文档重排序

        Args:
            query: 用户查询
            documents: 候选文档文本列表
            top_n: 返回前 N 个结果 (None 则返回全部)
            return_documents: 是否在结果中包含文档原文

        Returns:
            list[dict]: 排序后的结果列表, 每个包含:
                - index: 原始文档索引
                - relevance_score: 相关度分数 (0-1)
                - document (可选): 文档原文
        """
        if not documents:
            return []

        url = f"{self.base_url}/rerank"

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            # 按 relevance_score 降序排序
            results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            logger.debug(f"Reranked {len(documents)} docs → top score: "
                          f"{results[0]['relevance_score']:.4f}" if results else "empty")
            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Reranker API request failed: {e}")
            # 降级: 返回原始顺序
            return [{"index": i, "relevance_score": 1.0 - i * 0.01}
                    for i in range(len(documents))]
        except Exception as e:
            logger.error(f"Reranker error: {e}")
            return [{"index": i, "relevance_score": 1.0 - i * 0.01}
                    for i in range(len(documents))]

    def rerank_chunks(self, query: str, chunks: list[dict],
                      text_field: str = "content_with_weight",
                      top_n: int = None) -> list[dict]:
        """
        对 chunk 字典列表做重排序，返回重排后的 chunks

        Args:
            query: 用户查询
            chunks: chunk 字典列表
            text_field: 文本内容字段名
            top_n: 返回前 N 个

        Returns:
            list[dict]: 重排后的 chunk 列表 (每个 chunk 增加 rerank_score 字段)
        """
        if not chunks:
            return []

        # 提取文本
        documents = [ck.get(text_field, "") or " " for ck in chunks]

        # 过滤掉图谱上下文 chunk (不参与 rerank)
        graph_chunks = []
        text_indices = []
        text_docs = []

        for i, (ck, doc) in enumerate(zip(chunks, documents)):
            if ck.get("is_graph_context") or ck.get("doc_type_kwd") == "knowledge_graph":
                graph_chunks.append(ck)
            else:
                text_indices.append(i)
                text_docs.append(doc)

        if not text_docs:
            return list(chunks)

        # 调用 Reranker
        results = self.rerank(query, text_docs, top_n=top_n)

        # 映射回原始 chunk
        reranked_chunks = []
        for r in results:
            orig_idx = text_indices[r["index"]]
            ck = dict(chunks[orig_idx])
            ck["rerank_score"] = r["relevance_score"]
            reranked_chunks.append(ck)

        # 图谱上下文保持在最前面
        return graph_chunks + reranked_chunks
