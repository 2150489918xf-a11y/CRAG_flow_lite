"""
LLM 查询预处理增强
Step 1：用 LLM 提取核心关键词（含跨语言翻译），拼接到查询语句以提升 BM25 召回率
"""
import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from rag.llm.chat import ChatClient

logger = logging.getLogger(__name__)

# ==================== Prompt ====================

KEYWORD_SYSTEM = """你是一个查询优化助手。从用户问题中提取核心关键词，用于全文检索。

规则：
1. 提取 3-5 个最核心的关键词
2. 如果问题是中文，同时给出对应的英文翻译关键词
3. 如果问题是英文，同时给出对应的中文翻译关键词
4. 关键词应该是名词或名词短语，去掉虚词

输出严格使用 JSON 格式（不要添加任何其他内容）：
```json
{
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "translated": ["keyword1", "keyword2", "keyword3"]
}
```

示例：
问题："肺炎的症状有哪些"
```json
{"keywords": ["肺炎", "症状"], "translated": ["pneumonia", "symptoms"]}
```

问题："What are the side effects of aspirin?"
```json
{"keywords": ["side effects", "aspirin"], "translated": ["副作用", "阿司匹林"]}
```"""

KEYWORD_USER = """提取以下问题的核心关键词：

{question}"""


@dataclass
class EnhancedQuery:
    """增强后的查询"""
    original: str           # 原始查询
    keywords: list = field(default_factory=list)      # 原语言关键词
    translated: list = field(default_factory=list)     # 翻译后关键词
    enhanced_text: str = ""  # 增强后的查询文本（原始 + 翻译关键词）


class QueryEnhancer:
    """
    LLM 查询预处理增强器

    功能：
    1. 核心关键词提取 — 用廉价 LLM 提取 3-5 个关键词
    2. 跨语言翻译 — 同时输出原语言和目标语言关键词
    3. 查询增强 — 将翻译后的关键词拼接到原始查询
    """

    _CACHE_MAX = 256
    _CACHE_TTL = 600  # 10 分钟

    def __init__(self, chat_client: ChatClient = None):
        self.chat = chat_client or ChatClient()
        self._cache: OrderedDict[str, tuple[float, EnhancedQuery]] = OrderedDict()

    async def enhance(self, question: str) -> EnhancedQuery:
        """
        增强查询：提取关键词 + 跨语言翻译

        Returns:
            EnhancedQuery 包含原始查询、关键词、翻译词和增强文本
        """
        question = question.strip()
        if not question:
            return EnhancedQuery(original=question, enhanced_text=question)

        # 太短的查询不需要增强
        if len(question) < 2:
            return EnhancedQuery(original=question, enhanced_text=question)

        # 缓存命中
        cache_key = question.lower()
        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if time.time() - ts < self._CACHE_TTL:
                self._cache.move_to_end(cache_key)
                logger.info(f"QueryEnhancer cache HIT: {question[:30]}...")
                return cached
            else:
                del self._cache[cache_key]

        # 调用 LLM 提取关键词
        try:
            result = await self.chat.achat_json(
                KEYWORD_SYSTEM,
                KEYWORD_USER.format(question=question),
                temperature=0.1,
                max_tokens=256,
            )
        except Exception as e:
            logger.warning(f"QueryEnhancer LLM call failed: {e}")
            return EnhancedQuery(original=question, enhanced_text=question)

        keywords = result.get("keywords", [])
        translated = result.get("translated", [])

        # 构建增强文本：原始查询 + 翻译关键词
        enhanced_parts = [question]
        if translated:
            enhanced_parts.extend(translated)
        enhanced_text = " ".join(enhanced_parts)

        eq = EnhancedQuery(
            original=question,
            keywords=keywords,
            translated=translated,
            enhanced_text=enhanced_text,
        )

        logger.info(f"QueryEnhancer: '{question[:30]}' → keywords={keywords}, translated={translated}")

        # 写入缓存
        self._cache[cache_key] = (time.time(), eq)
        if len(self._cache) > self._CACHE_MAX:
            self._cache.popitem(last=False)

        return eq
