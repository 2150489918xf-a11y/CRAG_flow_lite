"""
CRAG 动态路由器 (Router) — 状态机总枢纽

在 4.4(Reranker精排) 和 4.5(GraphRAG) 之后、4.6(Prompt组装) 之前拦截，
基于 LLM 评估结果执行三路流转：

  🟢 Correct   → 直接放行，零额外开销
  🔴 Incorrect → 焦土政策，清空本地垃圾，全走外搜
  🟡 Ambiguous → 双管齐下，asyncio.gather 并发执行提炼+外搜

设计要点：
- 评估器在输出 Ambiguous 的同时一并输出 search_query（算子折叠，解决冲突4）
- 提炼和外搜并发执行，耗时降 50%（解决冲突4）
- 外搜结果统一伪装成 Virtual Chunk（解决冲突3）
"""
import asyncio
import logging
import time

from rag.crag.evaluator import CRAGEvaluator
from rag.crag.web_search import WebSearcher
from rag.crag.refiner import KnowledgeRefiner
from rag.llm.chat import ChatClient
from rag.settings import get_config

logger = logging.getLogger(__name__)


class CRAGRouter:
    """CRAG 动态路由控制台"""

    def __init__(self, chat_client: ChatClient = None):
        self.chat = chat_client or ChatClient()
        self.evaluator = CRAGEvaluator(self.chat)
        self.web_searcher = WebSearcher()
        self.refiner = KnowledgeRefiner(self.chat)

    async def route(self, question: str, local_chunks: list,
                    graph_context: str = "") -> dict:
        """
        CRAG 动态路由主入口

        接收 4.4 (Reranker) 和 4.5 (GraphRAG) 的结果，
        经 LLM 评估后，输出最终送给 4.6 Prompt 组装的纯净数据。

        Args:
            question: 用户问题
            local_chunks: Reranker 精排后的本地 chunks
            graph_context: GraphRAG 格式化 CSV 上下文

        Returns:
            dict: {
                "chunks": list[dict],       # 最终 chunks（可能含 Virtual Chunks）
                "graph_context": str,        # 最终图谱上下文（可能被清空）
                "crag_score": str,           # Correct/Incorrect/Ambiguous
                "crag_reason": str,          # 评判理由
                "crag_action": str,          # 执行的动作描述
                "latency_ms": int,           # CRAG 耗时
            }
        """
        t0 = time.time()

        # ========== Step 1: 联合评估 ==========
        logger.info(f"CRAG: evaluating context for: {question[:50]}...")
        eval_result = await self.evaluator.evaluate(question, local_chunks, graph_context)

        score = eval_result["score"]
        reason = eval_result["reason"]
        search_query = eval_result["search_query"]

        logger.info(f"CRAG verdict: {score} — {reason}")

        final_chunks = []
        final_graph = graph_context  # 图谱默认保留

        # ========== Step 2: 三路状态机流转 ==========

        if score == "Correct":
            # 🟢 完美命中：原生数据直接放行，零额外 API 调用
            final_chunks = local_chunks
            action = "PASS_THROUGH: 本地知识充足，直接放行"

        elif score == "Incorrect":
            # 🔴 幻觉熔断：焦土政策
            # 清空本地垃圾（包括图谱），防止 LLM 被误导产生幻觉
            final_graph = ""
            logger.info(f"CRAG: Incorrect → scorched earth, web searching: {search_query}")

            web_chunks = await self.web_searcher.search(search_query, top_k=3)

            if web_chunks:
                final_chunks = web_chunks
                action = f"SCORCHED_EARTH: 清空本地，外网召回 {len(web_chunks)} 条"
            else:
                # 外搜也失败了 → 降级为返回原始数据（总比空的好）
                final_chunks = local_chunks
                final_graph = graph_context
                action = "SCORCHED_EARTH_FALLBACK: 外搜失败，降级返回原始数据"
                logger.warning("CRAG: web search also failed, falling back to local")

        elif score == "Ambiguous":
            # 🟡 信息残缺：双管齐下（并发执行，解决冲突4）
            logger.info(f"CRAG: Ambiguous → parallel refine + web search: {search_query}")

            # asyncio.gather 并发：提炼本地 + 外网找补
            refine_task = self.refiner.refine(question, local_chunks)
            search_task = self.web_searcher.search(search_query, top_k=2)

            refined_chunks, web_chunks = await asyncio.gather(
                refine_task, search_task
            )

            final_chunks = refined_chunks + web_chunks
            action = (f"DUAL_AUGMENT: 提炼 {len(refined_chunks)} 条 + "
                       f"外搜 {len(web_chunks)} 条")

            if not final_chunks:
                # 都失败了 → 降级
                final_chunks = local_chunks
                action = "DUAL_AUGMENT_FALLBACK: 提炼和外搜均失败，降级返回原始数据"

        else:
            # 未知状态 → 安全默认
            final_chunks = local_chunks
            action = "UNKNOWN_FALLBACK"

        latency = int((time.time() - t0) * 1000)
        logger.info(f"CRAG completed: {action} ({latency}ms)")

        return {
            "chunks": final_chunks,
            "graph_context": final_graph,
            "crag_score": score,
            "crag_reason": reason,
            "crag_action": action,
            "latency_ms": latency,
        }
