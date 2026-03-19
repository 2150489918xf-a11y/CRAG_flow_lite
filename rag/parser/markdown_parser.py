"""
Markdown 解析器
"""
import logging
import re

logger = logging.getLogger(__name__)


def parse(filename, binary=None):
    """
    解析 Markdown 文件，返回 (sections, tables)
    按标题层级拆分段落
    """
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
