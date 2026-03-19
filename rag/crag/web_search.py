"""
CRAG 外网搜索适配器

调用 Tavily API 搜索，将结果伪装（Mock）成 RAGFlow Lite 标准 Chunk 格式，
实现底层架构对数据源的无感兼容（解决冲突3: 4.6 步对外网数据的 KeyError）。

也支持 DuckDuckGo 作为备选。
"""
import logging
from typing import Optional

import requests

from rag.settings import get_config

logger = logging.getLogger(__name__)


class WebSearcher:
    """外网搜索 + Virtual Chunk 伪装"""

    def __init__(self, api_key=None, provider="tavily"):
        cfg = get_config().get("crag", {})
        self.provider = provider or cfg.get("search_provider", "tavily")
        self.api_key = api_key or cfg.get("tavily_api_key", "")

    async def search(self, query: str, top_k: int = 3) -> list:
        """
        调用搜索引擎并返回伪装成标准 Chunk 格式的 Virtual Chunks

        Args:
            query: 搜索查询词
            top_k: 返回结果数

        Returns:
            list[dict]: 伪装成标准 ES chunk 格式的搜索结果
        """
        if self.provider == "tavily":
            return await self._tavily_search(query, top_k)
        elif self.provider == "duckduckgo":
            return await self._duckduckgo_search(query, top_k)
        else:
            logger.warning(f"Unknown search provider: {self.provider}")
            return []

    async def _tavily_search(self, query: str, top_k: int = 3) -> list:
        """Tavily API 搜索"""
        if not self.api_key:
            logger.warning("Tavily API key not configured, skipping web search")
            return []

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": top_k,
            "include_answer": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            virtual_chunks = []

            # Tavily 直接给出的总结答案
            if data.get("answer"):
                virtual_chunks.append(self._make_virtual_chunk(
                    idx=0,
                    title="Tavily AI 总结",
                    content=data["answer"],
                    url="tavily://answer",
                ))

            # 各条搜索结果
            for i, result in enumerate(data.get("results", [])):
                virtual_chunks.append(self._make_virtual_chunk(
                    idx=i + 1,
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    url=result.get("url", ""),
                ))

            logger.info(f"Tavily search '{query}': {len(virtual_chunks)} results")
            return virtual_chunks[:top_k + 1]

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

    async def _duckduckgo_search(self, query: str, top_k: int = 3) -> list:
        """DuckDuckGo 搜索 (备选，无需 API Key)"""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=top_k))

            virtual_chunks = []
            for i, r in enumerate(results):
                virtual_chunks.append(self._make_virtual_chunk(
                    idx=i,
                    title=r.get("title", ""),
                    content=r.get("body", ""),
                    url=r.get("href", ""),
                ))

            logger.info(f"DuckDuckGo search '{query}': {len(virtual_chunks)} results")
            return virtual_chunks

        except ImportError:
            logger.error("duckduckgo_search not installed: pip install duckduckgo-search")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    @staticmethod
    def _make_virtual_chunk(idx: int, title: str, content: str, url: str) -> dict:
        """
        将外网搜索结果伪装成 RAGFlow Lite 标准 Chunk 格式

        解决冲突3：外网数据直接传入 4.6 步不会触发 KeyError
        """
        return {
            "chunk_id": f"Web_Search_{idx:02d}",
            "id": f"web_{idx:02d}",
            "content_with_weight": f"【外网文献：{title}】\n{content}",
            "docnm_kwd": f"[Web] {title[:40]}",
            "doc_type_kwd": "web_search",
            "knowledge_graph_kwd": "web",
            "source_url": url,
            "similarity": 0.8,  # 给予中等偏高的默认分数
        }
