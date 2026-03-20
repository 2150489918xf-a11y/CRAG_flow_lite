"""
PDF 深度解析器 — 基于 RAGFlow 原版 DeepDoc 引擎
版面分析 + OCR + 表格识别 + XGBoost 文本拼接
"""
import logging
import os
import re

# HuggingFace 镜像（确保在 import deepdoc 前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logger = logging.getLogger(__name__)

# 全局单例 — DeepDoc 解析器初始化很重（加载 ONNX 模型），只初始化一次
_pdf_parser = None


def _get_deepdoc_parser():
    """懒加载 DeepDoc PDF 解析器（单例）"""
    global _pdf_parser
    if _pdf_parser is None:
        logger.info("Initializing DeepDoc RAGFlowPdfParser (first-time model download may take minutes)...")
        from deepdoc.parser.pdf_parser import RAGFlowPdfParser
        _pdf_parser = RAGFlowPdfParser()
        logger.info("DeepDoc RAGFlowPdfParser initialized successfully")
    return _pdf_parser


def _bbox_to_sections(bboxes):
    """
    将 DeepDoc 返回的 BBox 列表转换为 chunking.py 期望的 (sections, tables) 格式

    BBox 结构:
    {
        "text": "...",
        "layout_type": "title" | "text" | "table" | "figure" | ...,
        "page_number": 1,
        "x0": ..., "x1": ..., "top": ..., "bottom": ...,
    }

    sections: [(text, tag), ...]   — 供 naive_merge 合并
    tables:   [html_str, ...]      — 表格独立成块
    """
    sections = []
    tables = []

    for bbox in bboxes:
        text = bbox.get("text", "").strip()
        if not text:
            continue

        layout_type = bbox.get("layout_type", "text")
        page_num = bbox.get("page_number", 1)

        if layout_type == "table":
            # 表格独立成块
            tables.append(text)
        elif layout_type == "figure":
            # 图片描述作为普通文本
            if text and not text.startswith("<"):
                sections.append((text, f"figure_page_{page_num}"))
        else:
            # title / text / caption / etc. → 正文
            # 清理多余空白
            text = re.sub(r"[ \t]+", " ", text)
            sections.append((text, f"page_{page_num}"))



    return sections, tables


def parse(filename, binary=None):
    """
    使用 DeepDoc 深度视觉引擎解析 PDF

    Returns:
        (sections, tables)
        sections: [(text, tag), ...]
        tables: [html_string, ...]
    """
    try:
        parser = _get_deepdoc_parser()
    except Exception as e:
        logger.error(f"DeepDoc initialization failed, falling back to simple parser: {e}", exc_info=True)
        return _parse_simple(filename, binary)

    try:
        # 准备输入
        if binary:
            fnm = binary
        elif os.path.isfile(filename):
            with open(filename, "rb") as f:
                fnm = f.read()
        else:
            logger.error(f"PDF file not found: {filename}")
            return [], []

        # 获取包含版面信息的 BBox 列表（表格已作为 BBox 插入且包含 HTML）
        bboxes = parser.parse_into_bboxes(fnm, zoomin=3)

        return _bbox_to_sections(bboxes)

    except Exception as e:
        logger.error(f"DeepDoc parsing failed for {filename}: {e}", exc_info=True)
        logger.info("Falling back to simple PDF parser")
        return _parse_simple(filename, binary)


def _parse_simple(filename, binary=None):
    """
    简单 PDF 解析器（回退方案）
    基于 PyMuPDF (fitz) 进行纯文本提取
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("Neither DeepDoc nor PyMuPDF available for PDF parsing")
        return [], []

    try:
        if binary:
            doc = fitz.open(stream=binary, filetype="pdf")
        else:
            doc = fitz.open(filename)
    except Exception as e:
        logger.error(f"Failed to open PDF {filename}: {e}")
        return [], []

    sections = []
    tables = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if not text or not text.strip():
            continue

        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para = re.sub(r"[ \t]+", " ", para)
            para = re.sub(r"\n\s*\n", "\n", para)
            sections.append((para, f"page_{page_num + 1}"))

        try:
            page_tables = page.find_tables()
            if page_tables and page_tables.tables:
                for table in page_tables.tables:
                    try:
                        df = table.to_pandas()
                        if df is not None and not df.empty:
                            html = df.to_html(index=False)
                            tables.append(html)
                    except Exception:
                        pass
        except Exception:
            pass

    doc.close()
    return sections, tables
