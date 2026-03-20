"""
General 策略 (通用切分)
调用 naive_merge，适用于报告、文章等常规文档，按字数或标点直接截断。支持使用 Parent-Child 生成两级块结构。
"""
import copy
import hashlib
import re
from rag.nlp import naive_merge, rag_tokenizer

def _make_chunk_id(docnm, idx, content):
    raw = f"{docnm}_{idx}_{content[:50]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def chunk(filename, sections, tables, lang, parser_config):
    chunk_token_num = parser_config.get("chunk_token_num", 512)
    delimiter = parser_config.get("delimiter", "\n!?。；！？")
    
    use_parent_child = parser_config.get("use_parent_child", False)
    parent_token_num = parser_config.get("parent_token_num", 1024)
    child_delimiter = parser_config.get("child_delimiter", "。；！？")

    # 1. 整理段落 (移除为空的内容)
    sec_texts = []
    for sec in sections:
        if isinstance(sec, tuple):
            t = sec[0]
            pos = sec[1] if len(sec)>1 else ""
            if t and str(t).strip():
                sec_texts.append((str(t), pos))
        else:
            if sec and str(sec).strip():
                sec_texts.append((str(sec), ""))

    # 2. 合并文本块
    # 如果开启父子块，则以大词元合并为父块
    merge_budget = parent_token_num if use_parent_child else chunk_token_num
    merged_texts = naive_merge(sec_texts, merge_budget, delimiter)

    # 3. 构造字典返回格式
    import os
    docnm = os.path.basename(filename)
    base_doc = {
        "docnm_kwd": docnm,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", docnm)),
        "doc_type_kwd": "text",
    }
    
    eng = lang.lower().startswith("en")
    chunks = []
    chunk_idx = 0

    def _tokenize_fill(d, txt):
        d["content_with_weight"] = txt
        t = re.sub(r"</?(?:table|td|caption|tr|th)(?:\s[^<>]{0,12})?>", " ", txt)
        d["content_ltks"] = rag_tokenizer.tokenize(t)
        d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])

    for parent_text in merged_texts:
        if not parent_text.strip():
            continue
            
        pid = _make_chunk_id(docnm, chunk_idx, parent_text)
        
        if use_parent_child:
            # 存为 Parent
            p_chunk = copy.deepcopy(base_doc)
            p_chunk["id"] = pid
            p_chunk["chunk_type_kwd"] = "parent"
            _tokenize_fill(p_chunk, parent_text)
            chunks.append(p_chunk)

            # 切分为 Child
            split_pattern = f"([{re.escape(child_delimiter)}])"
            child_parts = re.split(split_pattern, parent_text)
            
            c_text = ""
            for part in child_parts:
                if not part: continue
                c_text += part
                if re.fullmatch(split_pattern, part) or len(c_text) >= chunk_token_num:
                    if c_text.strip():
                        c_chunk = copy.deepcopy(base_doc)
                        c_chunk["id"] = _make_chunk_id(docnm, chunk_idx, c_text)
                        c_chunk["chunk_type_kwd"] = "child"
                        c_chunk["parent_id_kwd"] = pid
                        c_chunk["mom_with_weight"] = parent_text # 溯源全文本
                        _tokenize_fill(c_chunk, c_text.strip())
                        chunks.append(c_chunk)
                        chunk_idx += 1
                    c_text = ""
            if c_text.strip():
                c_chunk = copy.deepcopy(base_doc)
                c_chunk["id"] = _make_chunk_id(docnm, chunk_idx, c_text)
                c_chunk["chunk_type_kwd"] = "child"
                c_chunk["parent_id_kwd"] = pid
                c_chunk["mom_with_weight"] = parent_text
                _tokenize_fill(c_chunk, c_text.strip())
                chunks.append(c_chunk)
                chunk_idx += 1
        else:
            # 单一粒度切分 Flat
            chunk_doc = copy.deepcopy(base_doc)
            chunk_doc["id"] = pid
            chunk_doc["chunk_type_kwd"] = "flat"
            _tokenize_fill(chunk_doc, parent_text)
            chunks.append(chunk_doc)
            chunk_idx += 1

    # 4. 表格块处理
    for tbl_txt in tables:
        if isinstance(tbl_txt, tuple) and len(tbl_txt) > 0:
            tbl_txt = tbl_txt[0]
        if isinstance(tbl_txt, list):
            tbl_txt = "\n".join(tbl_txt)
        if not tbl_txt or not str(tbl_txt).strip():
            continue
            
        t_chunk = copy.deepcopy(base_doc)
        t_chunk["id"] = _make_chunk_id(docnm, chunk_idx, tbl_txt)
        t_chunk["doc_type_kwd"] = "table"
        t_chunk["chunk_type_kwd"] = "flat"
        _tokenize_fill(t_chunk, str(tbl_txt))
        chunks.append(t_chunk)
        chunk_idx += 1

    return chunks
