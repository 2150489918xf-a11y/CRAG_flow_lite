import os
import re

lite_nlp = "rag/nlp/__init__.py"
ragflow_nlp = "ragflow/ragflow/rag/nlp/__init__.py"

with open(ragflow_nlp, "r", encoding="utf-8") as f:
    rf_code = f.read()

# Replace imports
rf_code = rf_code.replace("from common.token_utils import num_tokens_from_string", "# num_tokens_from_string will be injected")

# Remove RAGFlow's __all__
rf_code = re.sub(r"__all__ = \['rag_tokenizer'\]\n", "", rf_code)

# We want to keep Lite's header and tokenizer shim
lite_header = """\"\"\"
RAGFlow Lite NLP 模块 (Integrated with DeepDoc Merging Algorithms)
\"\"\"
import re
import copy
import logging
import random
from collections import Counter, defaultdict
import roman_numbers as r
from word2number import w2n
from cn2an import cn2an
from PIL import Image

import chardet

from rag.nlp.tokenizer import (
    tokenizer as rag_tokenizer,
    tokenize,
    fine_grained_tokenize,
    is_chinese,
    is_number,
    is_alphabet,
)

__all__ = ['rag_tokenizer', 'tokenize', 'fine_grained_tokenize', 'naive_merge', 'tree_merge', 'hierarchical_merge']

def num_tokens_from_string(text):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        chinese_count = sum(1 for ch in text if '\\u4e00' <= ch <= '\\u9fff')
        other_count = len(text) - chinese_count
        return int(chinese_count * 1.5 + other_count / 4)

def truncate(text, max_length):
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

"""

# Extract everything from `all_codecs` to the end from ragflow's nlp
match = re.search(r"(all_codecs\s*=.*)", rf_code, flags=re.DOTALL)
rf_body = match.group(1)

# Fix tokenize call in RAGFlow (orig: from . import rag_tokenizer; rag_tokenizer.tokenize)
# since we imported rag_tokenizer directly, we can strip "from . import rag_tokenizer"
rf_body = rf_body.replace("from . import rag_tokenizer", "")

with open("rag/nlp/__init__.py", "w", encoding="utf-8") as f:
    f.write(lite_header + rf_body)

print("Merged!")
