"""
文档分块引擎 (精简自 RAGFlow rag/app/naive.py)
核心职责：解析文档 → 按 token 控制分块 → 生成可索引的 chunk 字典
"""
import copy
import hashlib
import logging
import os
import re

from rag.nlp import (
    naive_merge, hierarchical_merge,
    tokenize_fn, tokenize_chunks, tokenize_table,
    num_tokens_from_string, find_codec, rag_tokenizer,
)
from rag.settings import get_rag_config

logger = logging.getLogger(__name__)

# 格式 → 解析器映射
PARSER_FACTORY = {}


def _ensure_parsers_registered():
    """懒注册解析器"""
    if PARSER_FACTORY:
        return
    PARSER_FACTORY.update({
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


def _get_parser(ext):
    """按扩展名获取解析函数"""
    _ensure_parsers_registered()
    import importlib

    entry = PARSER_FACTORY.get(ext.lower())
    if not entry:
        return None

    if ":" in entry:
        # 格式: "module.path:function_name"
        module_path, func_name = entry.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)
    else:
        # 格式: "module.path" → 调用 module.parse()
        mod = importlib.import_module(entry)
        return mod.parse


def _make_chunk_id(docnm, idx, content):
    """生成 chunk ID"""
    raw = f"{docnm}_{idx}_{content[:50]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def chunk(filename, binary=None, lang="Chinese", parser_config=None):
    """
    文档分块主函数

    Args:
        filename: 文件名或路径
        binary: 文件二进制内容 (可选)
        lang: 语言 ("Chinese" / "English")
        parser_config: 分块配置覆盖

    Returns:
        list[dict]: chunk 字典列表，每个包含:
            - id: 唯一标识
            - content_with_weight: 原始文本
            - content_ltks: 粗粒度分词
            - content_sm_ltks: 细粒度分词
            - docnm_kwd: 文件名
            - doc_type_kwd: "text" | "table"
            - chunk_type_kwd: "flat" | "parent" | "child"
            - parent_id_kwd: 父块 ID (仅 child 类型)
    """
    rag_cfg = get_rag_config()
    if parser_config:
        rag_cfg.update(parser_config)

    chunk_token_num = rag_cfg.get("chunk_token_num", 512)
    delimiter = rag_cfg.get("delimiter", "\n!?。；！？")
    use_parent_child = rag_cfg.get("use_parent_child", False)
    parent_token_num = rag_cfg.get("parent_token_num", 1024)
    child_token_num = rag_cfg.get("child_token_num", 256)

    ext = os.path.splitext(filename)[-1].lower()
    if not ext and filename.find(".") > 0:
        ext = "." + filename.rsplit(".", 1)[-1].lower()

    # 获取解析器
    parser_fn = _get_parser(ext)
    if parser_fn is None:
        logger.warning(f"No parser for format: {ext}, falling back to TXT")
        from rag.parser.other_parsers import parse_txt
        parser_fn = parse_txt

    # 如果没有 binary，从文件读取
    if binary is None and os.path.isfile(filename):
        with open(filename, "rb") as f:
            binary = f.read()

    # 解析文档
    try:
        result = parser_fn(filename, binary)
        if isinstance(result, tuple) and len(result) == 2:
            sections, tables = result
        else:
            sections = result if isinstance(result, list) else []
            tables = []
    except Exception as e:
        logger.error(f"Failed to parse {filename}: {e}")
        return []

    if not sections and not tables:
        logger.warning(f"No content extracted from {filename}")
        return []

    eng = lang.lower().startswith("en")
    docnm = os.path.basename(filename)

    # 基础 chunk 模板
    doc_template = {
        "docnm_kwd": docnm,
        "title_tks": rag_tokenizer.tokenize(
            re.sub(r"\.[a-zA-Z]+$", "", docnm)
        ),
        "title_sm_tks": rag_tokenizer.fine_grained_tokenize(
            rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", docnm))
        ),
        "doc_type_kwd": "text",
    }

    chunks = []

    if use_parent_child:
        # ===== 父子块模式 =====
        hierarchy = hierarchical_merge(sections, parent_token_num,
                                       child_token_num, delimiter)
        chunk_idx = 0
        for pi, group in enumerate(hierarchy):
            parent_text = group["parent_text"]
            children = group["children"]

            # 生成 Parent chunk
            parent_d = copy.deepcopy(doc_template)
            parent_id = _make_chunk_id(docnm, f"p{pi}", parent_text)
            parent_d["id"] = parent_id
            parent_d["chunk_type_kwd"] = "parent"
            tokenize_fn(parent_d, parent_text, eng)
            chunks.append(parent_d)

            # 生成 Child chunks
            for ci, child_text in enumerate(children):
                child_d = copy.deepcopy(doc_template)
                child_id = _make_chunk_id(docnm, f"c{chunk_idx}", child_text)
                child_d["id"] = child_id
                child_d["chunk_type_kwd"] = "child"
                child_d["parent_id_kwd"] = parent_id
                child_d["position_int"] = [chunk_idx]
                tokenize_fn(child_d, child_text, eng)
                chunks.append(child_d)
                chunk_idx += 1

        logger.info(f"Chunked {filename} (parent-child): "
                     f"{len(hierarchy)} parents, {chunk_idx} children")
    else:
        # ===== 平铺模式 (原逻辑) =====
        merged_chunks = naive_merge(sections, chunk_token_num, delimiter)
        chunks = tokenize_chunks(merged_chunks, doc_template, eng)
        for i, ck in enumerate(chunks):
            content = ck.get("content_with_weight", "")
            ck["id"] = _make_chunk_id(docnm, i, content)
            ck["chunk_type_kwd"] = "flat"

        logger.info(f"Chunked {filename}: {len(chunks)} chunks "
                     f"({len(merged_chunks)} text + {len(tables)} tables)")

    # 处理表格 (两种模式通用)
    if tables:
        table_chunks = tokenize_table(tables, doc_template, eng)
        for ti, tck in enumerate(table_chunks):
            tck["id"] = _make_chunk_id(docnm, f"t{ti}",
                                        tck.get("content_with_weight", ""))
            tck["chunk_type_kwd"] = "flat"
        chunks.extend(table_chunks)

    return chunks


def chunk_files(file_paths, lang="Chinese", parser_config=None, doc_id_prefix=""):
    """
    批量分块多个文件

    Args:
        file_paths: 文件路径列表
        lang: 语言
        parser_config: 分块配置
        doc_id_prefix: 文档 ID 前缀

    Returns:
        list[dict]: 所有文件的 chunk 列表
    """
    all_chunks = []
    for fpath in file_paths:
        doc_id = doc_id_prefix + hashlib.md5(fpath.encode()).hexdigest()[:16]
        chunks = chunk(fpath, lang=lang, parser_config=parser_config)
        for ck in chunks:
            ck["doc_id"] = doc_id
        all_chunks.extend(chunks)
    return all_chunks
