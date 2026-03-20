"""
CRAG 联合裁判器 (Corrective Retrieval Evaluator)

在精排(4.4)和图谱检索(4.5)之后，用 1 次 LLM 调用对全部召回结果做全局评估。
输出三态判定：Correct / Incorrect / Ambiguous
同时输出 search_query 供外搜使用（避免 Ambiguous 状态下的串行灾难）。

设计要点：
- 图谱数据与文本数据用 XML 标签物理隔离，避免注意力分散
- 单次 LLM 调用，不逐块评估，避免延迟爆炸
- 输出 JSON 含 score + reason + search_query
"""
import json
import logging
from typing import Optional

from rag.llm.base import BaseChatClient, get_chat_client

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM = """你是一个苛刻的知识核查裁判。你的任务是综合评估提供的【知识图谱】和【文本片段】能否准确、完整地解答【用户问题】。

请严格按照以下标准评估：
1. "Correct": 提供的知识包含明确、直接的答案依据，可以完整回答问题。
2. "Incorrect": 提供的知识完全无关，或根本无法回答该问题。
3. "Ambiguous": 知识部分相关，但缺失核心细节（如缺乏最新数据、具体数值、时间节点等）。

仅输出以下格式的 JSON（不要添加任何其他内容）：
```json
{
    "score": "Correct 或 Incorrect 或 Ambiguous",
    "reason": "简短的一句话理由",
    "search_query": "如果 score 为 Incorrect 或 Ambiguous，生成一个外部搜索引擎的精准短查询。如果 Correct 则为空字符串。"
}
```"""

EVALUATOR_USER = """【用户问题】：{question}

<KnowledgeGraph>
{graph_context}
</KnowledgeGraph>

<TextChunks>
{text_chunks}
</TextChunks>"""


class CRAGEvaluator:
    """CRAG 联合裁判器"""

    def __init__(self, chat_client: BaseChatClient = None):
        self.chat = chat_client or get_chat_client()

    async def evaluate(self, question: str, text_chunks: list,
                       graph_context: str = "") -> dict:
        """
        对精排后的文本 chunks 和图谱上下文做联合评估

        Args:
            question: 用户问题
            text_chunks: Reranker 精排后的 chunk 列表
            graph_context: GraphRAG 格式化的 CSV 上下文

        Returns:
            dict: {
                "score": "Correct" | "Incorrect" | "Ambiguous",
                "reason": "评判理由",
                "search_query": "外搜关键词 (Correct时为空)"
            }
        """
        # 构建文本片段字符串（带编号）
        chunks_str = "\n".join([
            f"[{i}] {c.get('content_with_weight', '')[:500]}"
            for i, c in enumerate(text_chunks)
            if c.get("doc_type_kwd") != "knowledge_graph"  # 跳过图谱 chunk
        ])

        if not chunks_str.strip() and not graph_context.strip():
            # 完全没有召回内容
            return {
                "score": "Incorrect",
                "reason": "未召回任何相关知识",
                "search_query": question,
            }

        user_prompt = EVALUATOR_USER.format(
            question=question,
            graph_context=graph_context if graph_context else "无图谱上下文",
            text_chunks=chunks_str if chunks_str else "无文本片段",
        )

        try:
            result = await self.chat.achat_json(
                EVALUATOR_SYSTEM, user_prompt,
                temperature=0.1, max_tokens=512,
            )
        except Exception as e:
            logger.error(f"CRAG evaluator failed: {e}")
            # 降级：默认 Ambiguous（最安全的选择）
            return {
                "score": "Ambiguous",
                "reason": f"评估器异常: {str(e)[:50]}",
                "search_query": question,
            }

        # 规范化输出
        score = result.get("score", "Ambiguous")
        if score not in ("Correct", "Incorrect", "Ambiguous"):
            score = "Ambiguous"

        return {
            "score": score,
            "reason": result.get("reason", ""),
            "search_query": result.get("search_query", question if score != "Correct" else ""),
        }
