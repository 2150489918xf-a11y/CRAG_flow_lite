"""
Manual 策略 (说明书/技术说明)
设备说明书、IT 文档等往往存在层级嵌套和长篇警告列表。通过 hierarchical_merge 保全编号嵌套，且对没有规律的部分执行 naive_merge 退化切割。
"""
import copy
import hashlib
import re

from rag.nlp import hierarchical_merge, bullets_category, naive_merge, rag_tokenizer

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

    sec_texts = []
    for sec in sections:
        if isinstance(sec, tuple):
            t = sec[0]
            pos = sec[1] if len(sec)>1 else ""
            if t and str(t).strip():
                sec_texts.append((str(t).strip(), pos))
        else:
            if sec and str(sec).strip():
                sec_texts.append((str(sec).strip(), ""))

    bull = bullets_category([t for t, _ in sec_texts])

    if bull >= 0:
        depth = parser_config.get("manual_depth", 2) # Manuals typically don't go as deep as laws
        merged_groups = hierarchical_merge(bull, [t for t, _ in sec_texts], depth)
        for group in merged_groups:
            if not group: continue
            concat_text = "\n".join(group)
            
            chunk_doc = copy.deepcopy(base_doc)
            chunk_doc["id"] = _make_chunk_id(docnm, chunk_idx, concat_text)
            chunk_doc["chunk_type_kwd"] = "flat"
            _tokenize_fill(chunk_doc, concat_text)
            chunks.append(chunk_doc)
            chunk_idx += 1
    else:
        chunk_token_num = parser_config.get("chunk_token_num", 512)
        delimiter = parser_config.get("delimiter", "\n!?。；！？")
        merged_texts = naive_merge(sec_texts, chunk_token_num, delimiter)
        for text in merged_texts:
            if not text.strip(): continue
            chunk_doc = copy.deepcopy(base_doc)
            chunk_doc["id"] = _make_chunk_id(docnm, chunk_idx, text)
            chunk_doc["chunk_type_kwd"] = "flat"
            _tokenize_fill(chunk_doc, text)
            chunks.append(chunk_doc)
            chunk_idx += 1

    for tbl_txt in tables:
        if isinstance(tbl_txt, tuple) and len(tbl_txt) > 0:
            tbl_txt = tbl_txt[0]
        if isinstance(tbl_txt, list):
            tbl_txt = "\n".join(tbl_txt)
        if not tbl_txt or not str(tbl_txt).strip(): continue
            
        t_chunk = copy.deepcopy(base_doc)
        t_chunk["id"] = _make_chunk_id(docnm, chunk_idx, tbl_txt)
        t_chunk["doc_type_kwd"] = "table"
        t_chunk["chunk_type_kwd"] = "flat"
        _tokenize_fill(t_chunk, str(tbl_txt))
        chunks.append(t_chunk)
        chunk_idx += 1

    return chunks
