"""
Presentation 策略 (幻灯片严苛切分)
以"页(Slide)"为绝对隔离边界，每一页的内容强行合并在一个独立的 Chunk 中，绝不跨页污染上下文。
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
    chunks = []
    chunk_idx = 0

    def _tokenize_fill(d, txt):
        d["content_with_weight"] = txt
        t = re.sub(r"</?(?:table|td|caption|tr|th)(?:\s[^<>]{0,12})?>", " ", txt)
        d["content_ltks"] = rag_tokenizer.tokenize(t)
        d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])

    # 尝试根据 pos 里的位置信息解析出页码并按页聚类
    # 格式通常包含坐标字典形式或者其它带 pn 的形式
    page_texts = {}
    
    for sec in sections:
        txt = ""
        pn = -1
        if isinstance(sec, tuple):
            txt = str(sec[0])
            pos_str = str(sec[1]) if len(sec)>1 else ""
            # 从 pos 提取页码 (例如匹配 'page_number': 1 或类似结构)
            m = re.search(r"'page_number'\s*:\s*(\d+)", pos_str)
            if not m:
                m = re.search(r"pn[=:_]?(\d+)", pos_str.lower())
            if m:
                pn = int(m.group(1))
        else:
            txt = str(sec)
            
        if not txt.strip(): continue
        
        # 兜底：若拿不到页码，则把它当成新的一页（自增）
        if pn == -1:
            pn = len(page_texts) + 9999
            
        if pn not in page_texts:
            page_texts[pn] = []
        page_texts[pn].append(txt)

    # 遍历每一页，合并为一个 Chunk
    for pn_key in sorted(page_texts.keys()):
        merged_txt = "\n".join(page_texts[pn_key])
        chunk_doc = copy.deepcopy(base_doc)
        chunk_doc["id"] = _make_chunk_id(docnm, chunk_idx, merged_txt)
        chunk_doc["chunk_type_kwd"] = "flat"
        _tokenize_fill(chunk_doc, merged_txt)
        chunks.append(chunk_doc)
        chunk_idx += 1

    # 表格由于通常自带隔离性，也独立作为 Chunk
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
