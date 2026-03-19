"""
RAGFlow Lite NLP 模块
提供分词、词权重、同义词、查询扩展等核心 NLP 功能
"""
import re
import copy
import logging

import chardet

from rag.nlp.tokenizer import (
    tokenizer as rag_tokenizer,
    tokenize,
    fine_grained_tokenize,
    is_chinese,
    is_number,
    is_alphabet,
)

__all__ = ['rag_tokenizer', 'tokenize', 'fine_grained_tokenize']

# 编码检测
all_codecs = [
    'utf-8', 'gb2312', 'gbk', 'utf_16', 'ascii', 'big5', 'big5hkscs',
    'cp037', 'cp437', 'cp850', 'cp852', 'cp855', 'cp857', 'cp858',
    'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866',
    'cp869', 'cp874', 'cp932', 'cp949', 'cp950',
    'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256',
    'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_kr',
    'gb18030', 'hz', 'iso2022_jp', 'iso2022_kr', 'latin_1',
    'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6',
    'iso8859_7', 'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_13',
    'iso8859_14', 'iso8859_15', 'shift_jis', 'utf_7',
]


def find_codec(blob):
    """检测文本编码"""
    detected = chardet.detect(blob[:1024])
    if detected['confidence'] > 0.5:
        if detected['encoding'] == "ascii":
            return "utf-8"

    for c in all_codecs:
        try:
            blob[:1024].decode(c)
            return c
        except Exception:
            pass
        try:
            blob.decode(c)
            return c
        except Exception:
            pass

    return "utf-8"


def num_tokens_from_string(text):
    """计算文本的 token 数量"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # fallback: 中文按 1.5 字/token, 英文按 4 字符/token
        chinese_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        other_count = len(text) - chinese_count
        return int(chinese_count * 1.5 + other_count / 4)


def truncate(text, max_length):
    """截断文本到指定 token 长度"""
    if not text:
        return text
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= max_length:
            return text
        return enc.decode(tokens[:max_length])
    except Exception:
        return text[:max_length * 4]


def tokenize_fn(d, txt, eng=False):
    """对文本进行分词并填充到 chunk 字典中"""
    d["content_with_weight"] = txt
    t = re.sub(r"</?(?:table|td|caption|tr|th)(?:\s[^<>]{0,12})?>", " ", txt)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])


def tokenize_chunks(chunks, doc, eng=False):
    """将文本块列表转为带分词信息的 chunk 字典列表"""
    res = []
    for ii, ck in enumerate(chunks):
        if not ck or not ck.strip():
            continue
        d = copy.deepcopy(doc)
        d["position_int"] = [ii]
        tokenize_fn(d, ck, eng)
        res.append(d)
    return res


def tokenize_table(tbls, doc, eng=False, batch_size=10):
    """将表格数据转为 chunk 字典列表"""
    res = []
    for rows in tbls:
        if not rows:
            continue
        if isinstance(rows, str):
            d = copy.deepcopy(doc)
            tokenize_fn(d, rows, eng)
            d["doc_type_kwd"] = "table"
            res.append(d)
            continue
        if isinstance(rows, list):
            de = "; " if eng else "；"
            for i in range(0, len(rows), batch_size):
                d = copy.deepcopy(doc)
                r = de.join(rows[i:i + batch_size])
                tokenize_fn(d, r, eng)
                d["doc_type_kwd"] = "table"
                res.append(d)
    return res


def naive_merge(sections, chunk_token_num=512, delimiter="\n!?。；！？"):
    """
    RAGFlow 核心分块算法：将段落按 token 数合并
    - sections: [(text, tag), ...] 形式的段落列表
    - chunk_token_num: 每个 chunk 的最大 token 数
    - delimiter: 句级分隔符
    """
    if not sections:
        return []

    chunks = []
    current_chunk = ""

    for sec in sections:
        if isinstance(sec, tuple):
            text = sec[0] if sec else ""
        elif isinstance(sec, str):
            text = sec
        else:
            continue

        if not text or not text.strip():
            continue

        # 如果当前 chunk 加上新段落还在 token 限制内，合并
        combined = current_chunk + "\n" + text if current_chunk else text
        if num_tokens_from_string(combined) <= chunk_token_num:
            current_chunk = combined
        else:
            # 当前 chunk 满了，保存并开始新 chunk
            if current_chunk:
                chunks.append(current_chunk)

            # 如果单段落就超过限制，按分隔符拆分
            if num_tokens_from_string(text) > chunk_token_num:
                # 按分隔符拆分
                split_pattern = f"([{re.escape(delimiter)}])"
                parts = re.split(split_pattern, text)
                sub_chunk = ""
                for part in parts:
                    if not part:
                        continue
                    test = sub_chunk + part
                    if num_tokens_from_string(test) <= chunk_token_num:
                        sub_chunk = test
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = part
                current_chunk = sub_chunk
            else:
                current_chunk = text

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
