"""
Q&A 策略 (问答切分)
适合处理 FAQ 表格与 FAQ 文本。精准提炼问答对，生成特殊的 Q-A 块，并在录入时只对 Question 进行语义 Token 提取，提供最大的检索命中率。
"""
import copy
import hashlib
import re
from rag.nlp import rag_tokenizer

def _make_chunk_id(docnm, idx, content):
    raw = f"{docnm}_{idx}_{content[:50]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def rmPrefix(txt):
    return re.sub(r"^(问题|答案|回答|user|assistant|Q|A|Question|Answer|问|答)[\t:： ]+", "", txt.strip(), flags=re.IGNORECASE)

def _build_qa_chunk(base_doc, q, a, lang, idx):
    eng = lang.lower().startswith("en")
    qprefix = "Question: " if eng else "问题："
    aprefix = "Answer: " if eng else "回答："
    
    chunk_doc = copy.deepcopy(base_doc)
    txt_content = f"{qprefix}{rmPrefix(q)}\t{aprefix}{rmPrefix(a)}"
    
    chunk_doc["id"] = _make_chunk_id(base_doc["docnm_kwd"], idx, txt_content)
    chunk_doc["content_with_weight"] = txt_content
    # 核心特色：仅对 Question(q) 部分进行分词索引 (content_ltks 和 content_sm_ltks)
    chunk_doc["content_ltks"] = rag_tokenizer.tokenize(q)
    chunk_doc["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(chunk_doc["content_ltks"])
    chunk_doc["chunk_type_kwd"] = "flat"
    return chunk_doc

def extract_qa_from_html_table(html_tbl):
    """从提取出的 HTML 表格中解析出 Q&A 列表"""
    import xml.etree.ElementTree as ET
    res = []
    try:
        # 简单补齐让它能被解析
        if not html_tbl.startswith("<table>"): html_tbl = "<table>" + html_tbl
        if not html_tbl.endswith("</table>"): html_tbl = html_tbl + "</table>"
        root = ET.fromstring(html_tbl)
        for tr in root.findall(".//tr"):
            tds = tr.findall(".//td")
            if len(tds) >= 2:
                q = "".join(tds[0].itertext()).strip()
                a = "".join(tds[1].itertext()).strip()
                if q and a and q.lower() not in ["question", "问题", "q"]:
                    res.append((q, a))
    except Exception:
        pass
    return res

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
    
    # 1. 优先从表格里智能抽取问答对 (非常适合 EXCEL / CSV 转换出来的 Table)
    for tbl_txt in tables:
        if isinstance(tbl_txt, tuple) and len(tbl_txt) > 0:
            tbl_txt = tbl_txt[0]
        if isinstance(tbl_txt, list):
            tbl_txt = "".join(tbl_txt)
        if not tbl_txt or not str(tbl_txt).strip():
            continue
            
        qa_pairs = extract_qa_from_html_table(str(tbl_txt))
        for q, a in qa_pairs:
            chunks.append(_build_qa_chunk(base_doc, q, a, lang, chunk_idx))
            chunk_idx += 1
            
    # 2. 从段落文字里硬匹配问答对
    # 如果有 Q: ... A: ... 的前缀结构
    current_q = ""
    current_a = ""
    
    q_pattern = re.compile(r"^(问题|Q|Question|问)[\t:： ]+(.*)", flags=re.IGNORECASE)
    a_pattern = re.compile(r"^(答案|回答|A|Answer|答)[\t:： ]+(.*)", flags=re.IGNORECASE)
    
    for sec in sections:
        txt = sec[0] if isinstance(sec, tuple) else str(sec)
        if not txt.strip(): continue
        
        # 尝试按行切割以便更精确识别
        lines = txt.split("\n")
        for line in lines:
            if not line.strip(): continue
            
            q_match = q_pattern.match(line)
            a_match = a_pattern.match(line)
            
            if q_match:
                # 若之前有没保存的，先保存
                if current_q and current_a:
                    chunks.append(_build_qa_chunk(base_doc, current_q, current_a, lang, chunk_idx))
                    chunk_idx += 1
                current_q = q_match.group(2)
                current_a = ""
            elif a_match:
                current_a = a_match.group(2)
            else:
                if current_q and not current_a:
                    current_q += "\n" + line
                elif current_q and current_a:
                    current_a += "\n" + line
                    
    # 收尾
    if current_q and current_a:
        chunks.append(_build_qa_chunk(base_doc, current_q, current_a, lang, chunk_idx))

    return chunks
