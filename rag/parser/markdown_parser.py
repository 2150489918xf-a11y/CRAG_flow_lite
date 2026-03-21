"""
Markdown 解析器
"""
import logging
import re
from typing import Optional

from rag.parser.base import BaseParser
from common.registry import parser_registry

logger = logging.getLogger(__name__)


@parser_registry.register(".md")
@parser_registry.register(".markdown")
class MarkdownParser(BaseParser):
    """Markdown 文档解析器"""

    name = "Markdown Parser"
    extensions = [".md", ".markdown"]

    def parse(self, filename: str, binary: Optional[bytes] = None):
        """解析 Markdown 文件，按标题层级拆分段落"""
        try:
            if binary:
                from rag.nlp import find_codec
                codec = find_codec(binary)
                text = binary.decode(codec)
            else:
                with open(filename, "r", encoding="utf-8") as f:
                    text = f.read()
        except Exception as e:
            logger.error(f"Failed to read MD {filename}: {e}")
            return [], []

        sections = []
        tables = []

        current_title = ""
        current_content = []

        for line in text.split("\n"):
            # 检测标题行
            heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
            if heading_match:
                # 保存前一个段落
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        tag = "title" if not current_title else "text"
                        if current_title:
                            content = current_title + "\n" + content
                        sections.append((content, tag))
                    current_content = []
                current_title = heading_match.group(2).strip()
                continue

            # 检测表格行
            if re.match(r"^\s*\|", line):
                current_content.append(line)
                continue

            current_content.append(line)

        # 处理最后一段
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                if current_title:
                    content = current_title + "\n" + content
                sections.append((content, "text"))

        return sections, tables


# 向后兼容
_instance = MarkdownParser()
parse = _instance.parse
