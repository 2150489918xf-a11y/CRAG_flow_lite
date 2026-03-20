"""
文档分块引擎主入口 (Router)
核心职责：调用提取器提取文本段落与表格 → 根据 parser_id 路由至不同的处理策略脚本
"""
import logging
import os

from rag.settings import get_rag_config

logger = logging.getLogger(__name__)

# 提取器映射 (File Extension -> Extractor)
EXTRACTOR_FACTORY = {}

# 策略脚本映射 (parser_id -> Strategy Script)
CHUNKER_FACTORY = {}

def _ensure_factories_registered():
    if EXTRACTOR_FACTORY:
        return
    EXTRACTOR_FACTORY.update({
        ".pdf": "rag.parser.pdf_parser",
        ".docx": "rag.parser.docx_parser",
        ".doc": "rag.parser.docx_parser",
        ".xlsx": "rag.parser.excel_parser",
        ".xls": "rag.parser.excel_parser",
        ".md": "rag.parser.markdown_parser",
        ".markdown": "rag.parser.markdown_parser",
        ".pptx": "rag.parser.other_parsers:parse_ppt",
        ".ppt": "rag.parser.other_parsers:parse_ppt",
        ".txt": "rag.parser.other_parsers:parse_txt",
        ".html": "rag.parser.other_parsers:parse_html",
        ".htm": "rag.parser.other_parsers:parse_html",
        ".json": "rag.parser.other_parsers:parse_json",
        ".csv": "rag.parser.other_parsers:parse_txt",
    })

    CHUNKER_FACTORY.update({
        "naive": "rag.app.naive:chunk",
        "qa": "rag.app.qa:chunk",
        "laws": "rag.app.laws:chunk",
        "one": "rag.app.one:chunk",
    })

def _get_extractor(ext):
    _ensure_factories_registered()
    import importlib
    entry = EXTRACTOR_FACTORY.get(ext.lower())
    if not entry:
        return None
    if ":" in entry:
        module_path, func_name = entry.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)
    else:
        mod = importlib.import_module(entry)
        return mod.parse

def _get_chunker(parser_id):
    _ensure_factories_registered()
    import importlib
    entry = CHUNKER_FACTORY.get(parser_id.lower())
    if not entry:
        logger.warning(f"Strategy '{parser_id}' not found, falling back to 'naive'")
        entry = CHUNKER_FACTORY["naive"]
    
    module_path, func_name = entry.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def chunk(filename, binary=None, lang="Chinese", parser_config=None):
    """
    文档分块主入口 Router
    1. 调用底层 Parser 抽取多模态数据结构 (sections, tables)
    2. 将抽取到的数据根据 parser_config.parser_id 转发给对应策略脚本处理
    """
    rag_cfg = get_rag_config()
    if parser_config:
        rag_cfg.update(parser_config)

    ext = os.path.splitext(filename)[-1].lower()
    if not ext and filename.find(".") > 0:
        ext = "." + filename.rsplit(".", 1)[-1].lower()

    # 1. 抽取结构化文本
    extractor_fn = _get_extractor(ext)
    if extractor_fn is None:
        logger.warning(f"No extractor for format: {ext}, falling back to TXT")
        from rag.parser.other_parsers import parse_txt
        extractor_fn = parse_txt

    if binary is None and os.path.isfile(filename):
        with open(filename, "rb") as f:
            binary = f.read()

    try:
        result = extractor_fn(filename, binary)
        if isinstance(result, tuple) and len(result) == 2:
            sections, tables = result
        else:
            sections = result if isinstance(result, list) else []
            tables = []
    except Exception as e:
        logger.error(f"Failed to parse {filename}: {e}", exc_info=True)
        return []

    if not sections and not tables:
        logger.warning(f"No content extracted from {filename}")
        return []

    # 2. 策略路由分发
    parser_id = rag_cfg.get("parser_id", "naive")
    chunker_fn = _get_chunker(parser_id)

    logger.info(f"Routing document {filename} to chunking strategy: {parser_id}")
    try:
        chunks = chunker_fn(filename, sections, tables, lang, rag_cfg)
        return chunks
    except Exception as e:
        logger.error(f"Chunking strategy {parser_id} failed for {filename}: {e}", exc_info=True)
        return []

