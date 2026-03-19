"""
GraphRAG 实体/关系提取器
使用 LLM 从文本 Chunk 中提取实体和关系三元组
"""
import logging
from typing import Optional
from dataclasses import dataclass, field

from rag.llm.chat import ChatClient

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """你是一个专业的知识图谱构建助手。从给定的文本中提取实体和关系。

输出严格使用以下 JSON 格式（不要添加任何其他内容）：
```json
{
  "entities": [
    {"name": "实体名称", "type": "实体类型", "description": "简短描述"}
  ],
  "relations": [
    {"source": "源实体名称", "target": "目标实体名称", "description": "关系描述"}
  ]
}
```

实体类型使用以下标准类别：
- PERSON（人物）
- ORGANIZATION（组织/公司）
- LOCATION（地点）
- EVENT（事件）
- PRODUCT（产品/技术）
- CONCEPT（概念/术语）
- DATE（日期/时间）
- OTHER（其他）

提取规则：
1. 只提取文本中明确提到的实体和关系
2. 实体名称使用原文中的表述
3. 关系描述要简洁明了
4. 如果文本中没有明显的实体或关系，返回空列表
"""

EXTRACTION_USER_TEMPLATE = """请从以下文本中提取实体和关系：

---
{text}
---"""


@dataclass
class Entity:
    name: str
    type: str
    description: str = ""
    pagerank: float = 0.0
    chunk_id: str = ""


@dataclass
class Relation:
    source: str
    target: str
    description: str = ""
    chunk_id: str = ""


@dataclass
class ExtractionResult:
    entities: list = field(default_factory=list)
    relations: list = field(default_factory=list)


class GraphExtractor:
    """LLM 驱动的图谱信息提取器"""

    def __init__(self, chat_client: ChatClient = None):
        self.chat = chat_client or ChatClient()

    def extract(self, text: str, chunk_id: str = "") -> ExtractionResult:
        """
        从单个文本块中提取实体和关系

        Args:
            text: 文本内容
            chunk_id: 来源 chunk ID

        Returns:
            ExtractionResult 包含实体和关系列表
        """
        if not text or not text.strip():
            return ExtractionResult()

        # 截断过长文本
        if len(text) > 4000:
            text = text[:4000]

        try:
            result = self.chat.chat_json(
                EXTRACTION_SYSTEM_PROMPT,
                EXTRACTION_USER_TEMPLATE.format(text=text),
                temperature=0.1,
            )
        except Exception as e:
            logger.error(f"Graph extraction failed for chunk {chunk_id}: {e}")
            return ExtractionResult()

        if not result:
            return ExtractionResult()

        # 解析实体
        entities = []
        for e in result.get("entities", []):
            if not isinstance(e, dict) or not e.get("name"):
                continue
            entities.append(Entity(
                name=e["name"].strip(),
                type=e.get("type", "OTHER").strip().upper(),
                description=e.get("description", "").strip(),
                chunk_id=chunk_id,
            ))

        # 解析关系
        relations = []
        for r in result.get("relations", []):
            if not isinstance(r, dict) or not r.get("source") or not r.get("target"):
                continue
            relations.append(Relation(
                source=r["source"].strip(),
                target=r["target"].strip(),
                description=r.get("description", "").strip(),
                chunk_id=chunk_id,
            ))

        logger.debug(f"Extracted {len(entities)} entities, {len(relations)} relations from chunk {chunk_id}")
        return ExtractionResult(entities=entities, relations=relations)

    def extract_batch(self, chunks: list, text_field="content_with_weight",
                      id_field="id") -> ExtractionResult:
        """
        批量提取多个 chunk 的实体和关系

        Args:
            chunks: chunk 字典列表
            text_field: 文本内容字段名
            id_field: ID 字段名

        Returns:
            合并的 ExtractionResult
        """
        all_entities = []
        all_relations = []

        for i, ck in enumerate(chunks):
            text = ck.get(text_field, "")
            chunk_id = ck.get(id_field, f"chunk_{i}")

            if not text or len(text.strip()) < 20:
                continue

            result = self.extract(text, chunk_id)
            all_entities.extend(result.entities)
            all_relations.extend(result.relations)

            if (i + 1) % 10 == 0:
                logger.info(f"  Graph extraction progress: {i + 1}/{len(chunks)} chunks")

        logger.info(f"Total extracted: {len(all_entities)} entities, {len(all_relations)} relations")
        return ExtractionResult(entities=all_entities, relations=all_relations)
