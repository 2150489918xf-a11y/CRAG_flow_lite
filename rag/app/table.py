"""
Table 策略 (复杂宽表结构切分)
针对那些非 Q&A 对但依然具有复杂结构的 Excel 或 HTML Table 数据。将整张 HTML 数据源封箱入独立 Table Chunk，以支持全量列式切片问答引擎的回调。
"""
import copy
import hashlib
import re

from rag.nlp import rag_tokenizer, naive_merge

def _make_chunk_id(docnm, idx, content):
    raw = f"{docnm}_{idx}_{content[:50]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def chunk(filename, sections, tables, lang, parser_config):
    import os
    docnm = os.path.basename(filename)
    base_doc = {
        "docnm_kwd": docnm,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", docnm)),
        "doc_type_kwd": "text",
    }
    chunks = []
    chunk_idx = 0

    def _tokenize_fill(d, txt):
        d["content_with_weight"] = txt
        t = re.sub(r"</?(?:table|td|caption|tr|th)(?:\s[^<>]{0,12})?>", " ", txt)
        d["content_ltks"] = rag_tokenizer.tokenize(t)
        d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])

    # 常规附带的纯文本按 naive 聚合
    sec_texts = [(sec[0], sec[1] if len(sec)>1 else "") if isinstance(sec, tuple) else (sec, "") for sec in sections if sec]
    text_parts = naive_merge(sec_texts, parser_config.get("chunk_token_num", 512), parser_config.get("delimiter", "\n!?。；！？"))
    
    for txt in text_parts:
        if not txt.strip(): continue
        chunk_doc = copy.deepcopy(base_doc)
        chunk_doc["id"] = _make_chunk_id(docnm, chunk_idx, txt)
        chunk_doc["chunk_type_kwd"] = "flat"
        _tokenize_fill(chunk_doc, txt)
        chunks.append(chunk_doc)
        chunk_idx += 1

    # 重头戏：表格专用聚合
    for tbl_txt in tables:
        if isinstance(tbl_txt, tuple) and len(tbl_txt) > 0:
            tbl_txt = tbl_txt[0]
        if isinstance(tbl_txt, list):
            tbl_txt = "\n".join(tbl_txt)
        if not tbl_txt or not str(tbl_txt).strip(): continue
            
        t_chunk = copy.deepcopy(base_doc)
        t_chunk["id"] = _make_chunk_id(docnm, chunk_idx, str(tbl_txt))
        t_chunk["doc_type_kwd"] = "table"
        t_chunk["chunk_type_kwd"] = "flat"
        _tokenize_fill(t_chunk, str(tbl_txt))
        chunks.append(t_chunk)
        chunk_idx += 1

    return chunks
