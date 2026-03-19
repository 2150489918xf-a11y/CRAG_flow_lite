"""
PDF 简易解析器
基于 PyMuPDF (fitz) 进行纯文本提取
"""
import logging
import re

logger = logging.getLogger(__name__)


def parse(filename, binary=None):
    """
    提取 PDF 文本，返回 [(text, tag), ...]
    """
    import fitz  # PyMuPDF

    try:
        if binary:
            doc = fitz.open(stream=binary, filetype="pdf")
        else:
            doc = fitz.open(filename)
    except Exception as e:
        logger.error(f"Failed to open PDF {filename}: {e}")
        return []

    sections = []
    tables = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if not text or not text.strip():
            continue

        # 按段落拆分
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # 清理多余空白
            para = re.sub(r"[ \t]+", " ", para)
            para = re.sub(r"\n\s*\n", "\n", para)
            sections.append((para, f"page_{page_num + 1}"))

        # 尝试提取表格 (PyMuPDF 基础表格提取)
        try:
            page_tables = page.find_tables()
            if page_tables and page_tables.tables:
                for table in page_tables.tables:
                    try:
                        df = table.to_pandas()
                        if df is not None and not df.empty:
                            # 转为 HTML 表格
                            html = df.to_html(index=False)
                            tables.append(html)
                    except Exception:
                        pass
        except Exception:
            pass

    doc.close()
    return sections, tables
