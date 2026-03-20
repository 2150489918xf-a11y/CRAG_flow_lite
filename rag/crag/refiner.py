"""
CRAG 知识提炼器 (Refiner)

Ambiguous 状态下，本地知识虽然不够但也不能全盘丢弃。
用 LLM 将冗余文本"脱水"，提炼核心事实，给外网搜索数据腾出 Prompt 空间。
输出同样伪装成标准 Virtual Chunk 格式。
"""
import logging

from rag.llm.base import BaseChatClient, get_chat_client

logger = logging.getLogger(__name__)

REFINE_SYSTEM = """你是一个精准的信息提炼专家。从冗长的文档中提取与用户问题相关的核心客观事实。

规则：
1. 去除所有废话、无关段落和重复内容
2. 用简洁的要点(Bullet points)输出
3. 只保留客观事实，不添加主观判断
4. 如果文档与问题毫无关系，输出"无相关事实"
"""

REFINE_USER = """用户问题：{question}

以下是需要提炼的文档内容：
{documents}"""


class KnowledgeRefiner:
    """Ambiguous 状态下的本地知识提炼器"""

    def __init__(self, chat_client: BaseChatClient = None):
        self.chat = chat_client or get_chat_client()

    async def refine(self, question: str, text_chunks: list) -> list:
        """
        将冗余的本地文本 chunk 提炼为精简的核心事实

        Args:
            question: 用户问题
            text_chunks: 本地召回的 chunk 列表

        Returns:
            list[dict]: 伪装成标准 Chunk 格式的提炼结果（通常只有 1 个）
        """
        # 拼接文档内容（控制长度）
        docs_text = "\n\n".join([
            c.get("content_with_weight", "")[:800]
            for c in text_chunks
            if c.get("doc_type_kwd") != "knowledge_graph"
        ])

        if not docs_text.strip():
            return []

        # 截断过长文本
        if len(docs_text) > 6000:
            docs_text = docs_text[:6000] + "\n...(截断)"

        try:
            refined_text = await self.chat.achat(
                REFINE_SYSTEM,
                REFINE_USER.format(question=question, documents=docs_text),
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error(f"Knowledge refine failed: {e}")
            # 降级：返回原始 chunks 的首段
            return [{
                "chunk_id": "Local_Refined_Fallback",
                "id": "refined_fallback",
                "content_with_weight": docs_text[:1000],
                "docnm_kwd": "[本地知识-原文]",
                "doc_type_kwd": "refined",
                "knowledge_graph_kwd": "refined",
                "similarity": 0.6,
            }]

        if not refined_text or refined_text.strip() == "无相关事实":
            return []

        # 伪装成标准 Virtual Chunk
        return [{
            "chunk_id": "Local_Refined_01",
            "id": "refined_01",
            "content_with_weight": f"【本地知识提炼事实】\n{refined_text}",
            "docnm_kwd": "[本地知识-提炼]",
            "doc_type_kwd": "refined",
            "knowledge_graph_kwd": "refined",
            "similarity": 0.7,
        }]
