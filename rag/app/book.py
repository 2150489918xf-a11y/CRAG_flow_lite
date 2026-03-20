"""
Book 策略 (书籍大部头切分)
调用 tree_merge，由于书籍大多拥有严密的树状目录（如篇、章、节），该算法会穷极可能的将父节点与其下文合并，保持整个长章节不被暴力腰斩。
"""
import copy
import hashlib
import re
import math

from rag.nlp import tree_merge, bullets_category, rag_tokenizer

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

    # 1. 段落整理
    sec_texts = []
    for sec in sections:
        if isinstance(sec, tuple):
            t = sec[0]
            if t and str(t).strip():
                sec_texts.append((str(t).strip(), ""))
        else:
            if sec and str(sec).strip():
                sec_texts.append((str(sec).strip(), ""))

    # 2. 识别目录节点标号规律
    bull = bullets_category([t for t, _ in sec_texts])

    # 3. 如果没识别到章节树规律，回退为 general 粗切; 若识别成功则触发 tree_merge 挂载
    if bull >= 0:
        depth = parser_config.get("book_depth", 3)
        merged_texts = tree_merge(bull, sec_texts, depth)
    else:
        from rag.nlp import naive_merge
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

    # 4. 表格整理
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
