"""
Shared pytest fixtures for RAGFlow Lite tests
"""
import os
import sys
import tempfile

import pytest

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== 文本 Fixtures ====================

SAMPLE_CHINESE = "RAGFlow 是一个基于深度文档理解的开源 RAG 引擎。它提供了简洁的 RAG 工作流，适用于任何规模的企业。"
SAMPLE_ENGLISH = "The quick brown fox jumps over the lazy dog. RAGFlow is a powerful retrieval engine."

SAMPLE_LEGAL = """\
第一条 本法旨在规范人工智能的开发和应用。
第二条 人工智能开发者应当遵守以下原则：
（一）安全可控原则
（二）公平透明原则
第三条 本法自2025年1月1日起施行。
"""

SAMPLE_QA = """\
Q: 什么是 RAGFlow？
A: RAGFlow 是一个基于深度文档理解的开源 RAG 引擎。

Question: How does chunking work?
Answer: It splits documents into smaller pieces for better retrieval.
"""


@pytest.fixture
def sample_chinese():
    return SAMPLE_CHINESE


@pytest.fixture
def sample_english():
    return SAMPLE_ENGLISH


@pytest.fixture
def sample_legal():
    return SAMPLE_LEGAL


@pytest.fixture
def sample_qa():
    return SAMPLE_QA


@pytest.fixture
def long_text():
    """生成一段足够长的文本，用于触发分块"""
    return "\n\n".join([
        "RAGFlow 是一个基于深度文档理解的开源 RAG 引擎。" * 60,
        "它提供了简洁的 RAG 工作流，适用于任何规模的企业。" * 60,
        "核心特性包括深度文档理解、混合检索、智能分块等。" * 60,
    ])


@pytest.fixture
def tmp_txt_file(long_text):
    """创建临时 txt 文件，测试后自动清理"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(long_text)
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def tmp_file_factory():
    """工厂 fixture：创建任意内容的临时文件"""
    paths = []

    def _create(content: str, suffix=".txt"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as f:
            f.write(content)
            paths.append(f.name)
            return f.name

    yield _create
    for p in paths:
        if os.path.exists(p):
            os.unlink(p)


# ==================== Mock 数据 Fixtures ====================

@pytest.fixture
def mock_entities():
    """构造模拟图谱实体"""
    from rag.graph.extractor import Entity
    return [
        Entity(name="微软", type="ORGANIZATION", description="全球最大科技公司"),
        Entity(name="OpenAI", type="ORGANIZATION", description="AI 研究公司"),
        Entity(name="ChatGPT", type="PRODUCT", description="大语言模型"),
        Entity(name="Satya Nadella", type="PERSON", description="微软CEO"),
        Entity(name="Sam Altman", type="PERSON", description="OpenAI CEO"),
    ]


@pytest.fixture
def mock_relations():
    """构造模拟图谱关系"""
    from rag.graph.extractor import Relation
    return [
        Relation(source="微软", target="OpenAI", description="投资了130亿美元"),
        Relation(source="OpenAI", target="ChatGPT", description="开发了"),
        Relation(source="Satya Nadella", target="微软", description="是CEO"),
        Relation(source="Sam Altman", target="OpenAI", description="是CEO"),
    ]


@pytest.fixture
def mock_extraction(mock_entities, mock_relations):
    """构造模拟图谱提取结果"""
    from rag.graph.extractor import ExtractionResult
    return ExtractionResult(entities=mock_entities, relations=mock_relations)
