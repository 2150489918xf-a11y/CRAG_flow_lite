"""
Paper 策略 (学术论文专属剔除分析)
学术论文依赖版面解析给出的 Layout 标签，针对性剔除页眉页脚与长串无用的参考文献，重点关照 Abstract 提供摘要特征绑定。
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

    # 基于版面过滤无用标签 (References往往占据巨量空间且干扰问答)
    usable_sec_texts = []
    
    for sec in sections:
        txt = ""
        layout_type = ""
        if isinstance(sec, tuple):
            txt = str(sec[0])
            pos = str(sec[1]) if len(sec)>1 else ""
            # 解构 layout_type@@[pos]
            parts = pos.split("@@", 1)
            if len(parts) >= 2:
                layout_type = parts[0].strip().lower()
        else:
            txt = str(sec)
        
        if not txt.strip(): continue
        
        # 摒弃参考文献与无关的 Header/Footer
        if "reference" in layout_type or layout_type in ("header", "footer"):
            continue
            
        # 若为 Abstract 或 Title 强化语义头部
        if layout_type == "abstract":
            txt = f"[ABSTRACT] {txt}"
        elif layout_type == "title":
            txt = f"[TITLE] {txt}"
            
        usable_sec_texts.append((txt, ""))

    merged_texts = naive_merge(usable_sec_texts, parser_config.get("chunk_token_num", 512), parser_config.get("delimiter", "\n!?。；！？"))
    
    for txt in merged_texts:
        if not txt.strip(): continue
        chunk_doc = copy.deepcopy(base_doc)
        chunk_doc["id"] = _make_chunk_id(docnm, chunk_idx, txt)
        chunk_doc["chunk_type_kwd"] = "flat"
        _tokenize_fill(chunk_doc, txt)
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
