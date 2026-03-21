"""
解析器抽象基类 (BaseParser)

所有文档解析器都应继承此基类并实现 parse() 方法。
通过 @parser_registry.register(".ext") 装饰器注册到全局注册器。

扩展示例:

    from rag.parser.base import BaseParser
    from common.registry import parser_registry

    @parser_registry.register(".xyz")
    class XYZParser(BaseParser):
        name = "XYZ Parser"
        extensions = [".xyz"]

        def parse(self, filename, binary=None):
            sections = [("parsed text", "text")]
            tables = []
            return sections, tables

Section 格式:
    sections: list[tuple[str, str]]
        每个元素为 (text_content, tag)
        tag 取值: "text" | "title" | "table" | "page_N" | "slide_N" 等

Tables 格式:
    tables: list[str]
        每个元素为 HTML 表格字符串
"""
from abc import ABC, abstractmethod
from typing import Optional


class BaseParser(ABC):
    """文档解析器抽象基类"""

    # 子类应设置
    name: str = "BaseParser"
    extensions: list[str] = []  # 支持的文件扩展名, e.g. [".pdf", ".PDF"]

    @abstractmethod
    def parse(self, filename: str, binary: Optional[bytes] = None) -> tuple[list, list]:
        """
        解析文档，返回结构化数据

        Args:
            filename: 文件路径 (用于推断格式 / 读取文件)
            binary:   文件二进制内容 (可选，若提供则优先使用)

        Returns:
            (sections, tables)
            - sections: list[tuple[str, str]] — 文本段落 + 标签
            - tables:   list[str]             — HTML 表格字符串
        """
        ...

    def __repr__(self) -> str:
        exts = ", ".join(self.extensions) if self.extensions else "N/A"
        return f"{self.name}(extensions=[{exts}])"


class FunctionParser(BaseParser):
    """
    函数包装器 — 将已有的 parse(filename, binary) 函数包装为 BaseParser 实例。
    用于兼容现有的函数式解析器，无需改写为类。

    Usage:
        wrapped = FunctionParser(parse_fn, name="DOCX Parser", extensions=[".docx"])
        sections, tables = wrapped.parse("test.docx", binary)
    """

    def __init__(self, fn, name: str = "FunctionParser", extensions: list[str] = None):
        self.fn = fn
        self.name = name
        self.extensions = extensions or []

    def parse(self, filename, binary=None):
        return self.fn(filename, binary)
