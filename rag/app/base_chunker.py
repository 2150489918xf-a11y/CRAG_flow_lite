"""
分块策略抽象基类 (BaseChunker)

所有分块策略都应继承此基类并实现 chunk() 方法。
通过 @chunker_registry.register("name") 装饰器注册到全局注册器。

扩展示例:

    from rag.app.base_chunker import BaseChunker
    from common.registry import chunker_registry

    @chunker_registry.register("my_strategy")
    class MyChunker(BaseChunker):
        name = "My Custom Chunker"

        def chunk(self, filename, sections, tables, lang, parser_config):
            chunks = []
            for sec in self.iter_sections(sections):
                doc = self.make_base_doc(filename)
                doc["id"] = self.make_id(filename, len(chunks), sec)
                self.tokenize_fill(doc, sec)
                chunks.append(doc)
            return chunks

Section 格式 (来自 Parser):
    sections: list[tuple[str, str]]  — (text_content, tag)
    tables:   list[str]              — HTML 表格字符串

Chunk 输出格式:
    list[dict] — 每个 dict 至少包含:
        - id:                  str   — 唯一 ID
        - content_with_weight: str   — 原始文本
        - content_ltks:        str   — 分词后的 token 串
        - content_sm_ltks:     str   — 细粒度分词
        - docnm_kwd:           str   — 文档文件名
        - doc_type_kwd:        str   — "text" | "table"
        - chunk_type_kwd:      str   — "flat" | "parent" | "child"
"""
import copy
import hashlib
import os
import re
from abc import ABC, abstractmethod
from typing import Optional

from rag.nlp import rag_tokenizer


class BaseChunker(ABC):
    """分块策略抽象基类"""

    # 子类应设置
    name: str = "BaseChunker"

    @abstractmethod
    def chunk(self, filename: str, sections: list, tables: list,
              lang: str, parser_config: dict) -> list[dict]:
        """
        将解析器输出的 (sections, tables) 分块为检索用 chunk 列表

        Args:
            filename:       文件路径
            sections:       文本段落列表 [(text, tag), ...]
            tables:         HTML 表格列表 [html_str, ...]
            lang:           语言 ("Chinese" | "English" | ...)
            parser_config:  分块配置参数

        Returns:
            list[dict] — chunk 字典列表
        """
        ...

    # ── 通用工具方法 ──

    @staticmethod
    def make_id(docnm: str, idx: int, content: str) -> str:
        """生成 chunk ID (MD5)"""
        raw = f"{docnm}_{idx}_{content[:50]}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def make_base_doc(filename: str) -> dict:
        """创建 chunk 基础字典（包含文件名和标题分词）"""
        docnm = os.path.basename(filename)
        return {
            "docnm_kwd": docnm,
            "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", docnm)),
            "doc_type_kwd": "text",
        }

    @staticmethod
    def tokenize_fill(doc: dict, text: str):
        """对 chunk 文本进行分词填充"""
        doc["content_with_weight"] = text
        t = re.sub(r"</?(?:table|td|caption|tr|th)(?:\s[^<>]{0,12})?>", " ", text)
        doc["content_ltks"] = rag_tokenizer.tokenize(t)
        doc["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(doc["content_ltks"])

    @staticmethod
    def iter_sections(sections: list) -> list[str]:
        """从 sections 中提取纯文本列表"""
        texts = []
        for sec in sections:
            if isinstance(sec, tuple):
                t = sec[0]
                if t and str(t).strip():
                    texts.append(str(t).strip())
            else:
                if sec and str(sec).strip():
                    texts.append(str(sec).strip())
        return texts

    @staticmethod
    def iter_section_pairs(sections: list) -> list[tuple[str, str]]:
        """从 sections 中提取 (text, tag) 对"""
        pairs = []
        for sec in sections:
            if isinstance(sec, tuple):
                t = sec[0]
                pos = sec[1] if len(sec) > 1 else ""
                if t and str(t).strip():
                    pairs.append((str(t), pos))
            else:
                if sec and str(sec).strip():
                    pairs.append((str(sec), ""))
        return pairs

    def process_tables(self, tables: list, base_doc: dict, chunk_idx: int) -> tuple[list[dict], int]:
        """处理表格块（通用逻辑）"""
        chunks = []
        for tbl_txt in tables:
            if isinstance(tbl_txt, tuple) and len(tbl_txt) > 0:
                tbl_txt = tbl_txt[0]
            if isinstance(tbl_txt, list):
                tbl_txt = "\n".join(tbl_txt)
            if not tbl_txt or not str(tbl_txt).strip():
                continue

            t_chunk = copy.deepcopy(base_doc)
            t_chunk["id"] = self.make_id(base_doc["docnm_kwd"], chunk_idx, tbl_txt)
            t_chunk["doc_type_kwd"] = "table"
            t_chunk["chunk_type_kwd"] = "flat"
            self.tokenize_fill(t_chunk, str(tbl_txt))
            chunks.append(t_chunk)
            chunk_idx += 1

        return chunks, chunk_idx

    def __repr__(self) -> str:
        return f"{self.name}()"
