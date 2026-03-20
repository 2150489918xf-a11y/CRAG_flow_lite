"""
Laws 策略 (法律条文切分)
调用 hierarchical_merge，在解析法律条文、规章制度、合同等具备明显章节结构的文本时，保留上下文树结构，避免切分断层。
"""
import copy
import hashlib
import re

from rag.nlp import hierarchical_merge, bullets_category, rag_tokenizer

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

    # 1. 整理段落 (移除为空的内容)
    sec_texts = []
    for sec in sections:
        if isinstance(sec, tuple):
            t = sec[0]
            if t and str(t).strip():
                sec_texts.append(str(t).strip())
        else:
            if sec and str(sec).strip():
                sec_texts.append(str(sec).strip())

    # 2. 识别编号规律 (bull)
    bull = bullets_category(sec_texts)

    # 3. 如果成功识别到了法律条款树层级 (bull >= 0)，进行深度合并
    if bull >= 0:
        # 默认使用 3 层深度
        depth = parser_config.get("laws_depth", 3)
        # RAGFlow hierarchical_merge expect list of tuples (t, pos) but handles mix, but it's safer to pass flat strings
        merged_groups = hierarchical_merge(bull, sec_texts, depth)
        
        # hierarchical_merge 返回的是 List[List[text]] 的组合形式（带层次标题上下文）
        for group in merged_groups:
            # group 为包含了所有祖先节点标题到具体条款的文本列表
            if not group: continue
            concat_text = "\n".join(group)
            
            chunk_doc = copy.deepcopy(base_doc)
            chunk_doc["id"] = _make_chunk_id(docnm, chunk_idx, concat_text)
            chunk_doc["chunk_type_kwd"] = "flat"
            _tokenize_fill(chunk_doc, concat_text)
            chunks.append(chunk_doc)
            chunk_idx += 1
            
    else:
        # 如果未识别到法律条款层级前缀，降级为 naive_merge
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
