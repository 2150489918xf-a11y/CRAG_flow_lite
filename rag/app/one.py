"""
One 策略 (全篇切分)
极端的合并策略：不进行任何字数限制与切分判断，将一整份文档连带表格拼装成一个包含全文的超级大块。
常用于基于整本小册子或短文的全文直接总结请求场景。
"""
import copy
import hashlib
import re

from rag.nlp import rag_tokenizer

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
    
    full_text_parts = []
    
    # 将所有文本聚合
    for sec in sections:
        txt = sec[0] if isinstance(sec, tuple) else str(sec)
        if txt and txt.strip():
            full_text_parts.append(txt.strip())
            
    # 将所有 Table 聚合在末尾
    for tbl_txt in tables:
        if isinstance(tbl_txt, tuple) and len(tbl_txt) > 0:
            tbl_txt = tbl_txt[0]
        if isinstance(tbl_txt, list):
            tbl_txt = "\n".join(tbl_txt)
        if tbl_txt and str(tbl_txt).strip():
            full_text_parts.append(str(tbl_txt).strip())
            
    full_content = "\n\n".join(full_text_parts)
    if not full_content.strip():
        return []

    def _tokenize_fill(d, txt):
        d["content_with_weight"] = txt
        t = re.sub(r"</?(?:table|td|caption|tr|th)(?:\s[^<>]{0,12})?>", " ", txt)
        d["content_ltks"] = rag_tokenizer.tokenize(t)
        d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])

    chunk_doc = copy.deepcopy(base_doc)
    chunk_doc["id"] = _make_chunk_id(docnm, 0, full_content)
    chunk_doc["chunk_type_kwd"] = "flat"
    _tokenize_fill(chunk_doc, full_content)
    
    return [chunk_doc]
