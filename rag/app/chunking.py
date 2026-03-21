"""
文档分块引擎主入口 (Router)
核心职责：调用提取器提取文本段落与表格 → 根据 parser_id 路由至不同的处理策略脚本

扩展方式:
  # 添加新的解析器 (parser):
  from common.registry import parser_registry

  @parser_registry.register(".xyz")
  def parse(filename, binary):
      ...
      return sections, tables

  # 添加新的分块策略 (chunker):
  from common.registry import chunker_registry

  @chunker_registry.register("my_strategy")
  def chunk(filename, sections, tables, lang, parser_config):
      ...
      return [{"content_with_weight": ..., "id": ...}, ...]
"""
import logging
import os

from rag.settings import get_rag_config
from common.registry import parser_registry, chunker_registry

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
#  内置解析器注册（首次调用时懒加载）
# ══════════════════════════════════════════

_builtin_registered = False


def _register_builtins():
    """注册所有内置解析器和分块策略到全局注册器"""
    global _builtin_registered
    if _builtin_registered:
        return
    _builtin_registered = True

    import importlib

    # ── 解析器映射 (File Extension → module:func) ──
    _PARSER_MAP = {
        ".pdf": "rag.parser.pdf_parser:parse",
        ".docx": "rag.parser.docx_parser:parse",
        ".doc": "rag.parser.docx_parser:parse",
        ".xlsx": "rag.parser.excel_parser:parse",
        ".xls": "rag.parser.excel_parser:parse",
        ".md": "rag.parser.markdown_parser:parse",
        ".markdown": "rag.parser.markdown_parser:parse",
        ".pptx": "rag.parser.other_parsers:parse_ppt",
        ".ppt": "rag.parser.other_parsers:parse_ppt",
        ".txt": "rag.parser.other_parsers:parse_txt",
        ".html": "rag.parser.other_parsers:parse_html",
        ".htm": "rag.parser.other_parsers:parse_html",
        ".json": "rag.parser.other_parsers:parse_json",
        ".csv": "rag.parser.other_parsers:parse_txt",
    }

    # ── 分块策略映射 (parser_id → module:func) ──
    _CHUNKER_MAP = {
        "naive": "rag.app.naive:chunk",
        "qa": "rag.app.qa:chunk",
        "laws": "rag.app.laws:chunk",
        "one": "rag.app.one:chunk",
        "book": "rag.app.book:chunk",
        "paper": "rag.app.paper:chunk",
        "presentation": "rag.app.presentation:chunk",
        "table": "rag.app.table:chunk",
        "manual": "rag.app.manual:chunk",
    }

    def _lazy_loader(entry_str):
        """创建一个懒加载函数包装器"""
        _cached = {}
        def loader(*args, **kwargs):
            if "fn" not in _cached:
                module_path, func_name = entry_str.rsplit(":", 1)
                mod = importlib.import_module(module_path)
                _cached["fn"] = getattr(mod, func_name)
            return _cached["fn"](*args, **kwargs)
        return loader

    for ext, entry in _PARSER_MAP.items():
        if not parser_registry.has(ext):
            parser_registry.register(ext)(_lazy_loader(entry))

    for pid, entry in _CHUNKER_MAP.items():
        if not chunker_registry.has(pid):
            chunker_registry.register(pid)(_lazy_loader(entry))

    logger.debug(f"Registered {len(_PARSER_MAP)} parsers, {len(_CHUNKER_MAP)} chunkers")


# ══════════════════════════════════════════
#  分块主入口
# ══════════════════════════════════════════

def chunk(filename, binary=None, lang="Chinese", parser_config=None):
    """
    文档分块主入口 Router
    1. 调用底层 Parser 抽取多模态数据结构 (sections, tables)
    2. 将抽取到的数据根据 parser_config.parser_id 转发给对应策略脚本处理
    """
    _register_builtins()

    rag_cfg = get_rag_config()
    if parser_config:
        rag_cfg.update(parser_config)

    ext = os.path.splitext(filename)[-1].lower()
    if not ext and filename.find(".") > 0:
        ext = "." + filename.rsplit(".", 1)[-1].lower()

    # 1. 抽取结构化文本
    if parser_registry.has(ext):
        extractor_fn = parser_registry.get(ext)
    else:
        logger.warning(f"No parser for format: {ext}, falling back to TXT")
        extractor_fn = parser_registry.get(".txt")

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

    # 2. 策略路由分发 (First-Level Routing)
    parser_id = rag_cfg.get("parser_id", "naive")

    # 智能特例 (Smart Exception)：PPT 格式强行应用 presentation 切片策略
    if ext in [".ppt", ".pptx"]:
        parser_id = "presentation"
        logger.info(f"Smart Exception applied: {ext} format auto-routed to 'presentation' strategy.")

    if chunker_registry.has(parser_id):
        chunker_fn = chunker_registry.get(parser_id)
    else:
        logger.warning(f"Strategy '{parser_id}' not found, falling back to 'naive'")
        chunker_fn = chunker_registry.get("naive")

    logger.info(f"Routing document {filename} to chunking strategy: {parser_id}")
    try:
        chunks = chunker_fn(filename, sections, tables, lang, rag_cfg)
        return chunks
    except Exception as e:
        logger.error(f"Chunking strategy {parser_id} failed for {filename}: {e}", exc_info=True)
        return []
