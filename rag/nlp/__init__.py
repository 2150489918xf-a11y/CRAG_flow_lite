"""
RAGFlow Lite NLP 模块 — 公共 API 入口
所有函数拆分到子模块，此文件仅负责统一导出。
"""
from rag.nlp.tokenizer import (
    tokenizer as rag_tokenizer,
    tokenize as _raw_tokenize,
    fine_grained_tokenize,
    is_chinese,
    is_number,
    is_alphabet,
)

# text_utils
from rag.nlp.text_utils import (
    num_tokens_from_string,
    truncate,
    all_codecs,
    find_codec,
)

# bullet / 标题检测
from rag.nlp.bullet import (
    QUESTION_PATTERN,
    BULLET_PATTERN,
    has_qbullet,
    index_int,
    qbullets_category,
    random_choices,
    not_bullet,
    bullets_category,
    is_english,
    is_chinese,
    remove_contents_table,
    make_colon_as_title,
    title_frequency,
    not_title,
    docx_question_level,
    extract_between,
    get_delimiters,
)

# chunk tokenize
from rag.nlp.chunk_tokenize import (
    tokenize,
    split_with_pattern,
    tokenize_chunks,
    doc_tokenize_chunks_with_images,
    tokenize_chunks_with_images,
    tokenize_table,
    attach_media_context,
    append_context2table_image4pdf,
    add_positions,
)

# merge
from rag.nlp.merge import (
    tree_merge,
    hierarchical_merge,
    naive_merge,
    naive_merge_with_images,
    naive_merge_docx,
    concat_img,
    Node,
)

__all__ = [
    "rag_tokenizer", "tokenize", "fine_grained_tokenize",
    "is_chinese", "is_number", "is_alphabet",
    "num_tokens_from_string", "truncate", "all_codecs", "find_codec",
    "QUESTION_PATTERN", "BULLET_PATTERN",
    "has_qbullet", "index_int", "qbullets_category",
    "random_choices", "not_bullet", "bullets_category",
    "is_english", "remove_contents_table", "make_colon_as_title",
    "title_frequency", "not_title", "docx_question_level",
    "extract_between", "get_delimiters",
    "tokenize", "split_with_pattern", "tokenize_chunks",
    "doc_tokenize_chunks_with_images", "tokenize_chunks_with_images",
    "tokenize_table", "attach_media_context", "append_context2table_image4pdf",
    "add_positions",
    "tree_merge", "hierarchical_merge", "naive_merge",
    "naive_merge_with_images", "naive_merge_docx",
    "concat_img", "Node",
]
