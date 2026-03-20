import os
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.app import chunking
import json

def test_pipeline():
    # 1. Create a dummy txt document
    dummy_text = """
第一条 本法旨在测试 hierarchical_merge 的法律条文处理能力。
第二条 这是第二条，包含一些细则：
（一）第一项细则
（二）第二项细则

Q: 什么是 RAGFlow 路由策略？
A: 路由策略就是将不同场景文档交给不同的脚本。

Question: How does Q&A work?
Answer: It extracts question and answer pairs!
    """
    
    with open("dummy_test.txt", "w", encoding="utf-8") as f:
        f.write(dummy_text)
        
    print("=== Testing NAIVE Strategy (with Parent-Child) ===")
    cfg_naive = {
        "parser_id": "naive",
        "use_parent_child": True,
        "parent_token_num": 100,
        "child_token_num": 50,
        "chunk_token_num": 50,
        "child_delimiter": "。"
    }
    chunks_naive = chunking.chunk("dummy_test.txt", lang="Chinese", parser_config=cfg_naive)
    for c in chunks_naive:
        print(f"[{c.get('chunk_type_kwd')}] {c['content_with_weight'][:60]}...")
        if "parent_id_kwd" in c:
            print(f"  -> parent_id: {c['parent_id_kwd']}")
            print(f"  -> mom_with_weight: {c['mom_with_weight'][:40]}...")
            
    print("\n=== Testing Q&A Strategy ===")
    cfg_qa = {
        "parser_id": "qa"
    }
    chunks_qa = chunking.chunk("dummy_test.txt", lang="Chinese", parser_config=cfg_qa)
    for c in chunks_qa:
        print(f"[QA Chunk] {c['content_with_weight'][:60]}...")
        print(f"  -> Index(content_ltks) Tokens: {len(c['content_ltks'])}")
        
    print("\n=== Testing LAWS Strategy ===")
    cfg_laws = {
        "parser_id": "laws"
    }
    chunks_laws = chunking.chunk("dummy_test.txt", lang="Chinese", parser_config=cfg_laws)
    for c in chunks_laws:
        print(f"[LAWS Chunk] {repr(c['content_with_weight'][:80])}...")
        
    print("\n=== Testing ONE Strategy ===")
    cfg_one = {
        "parser_id": "one"
    }
    chunks_one = chunking.chunk("dummy_test.txt", lang="Chinese", parser_config=cfg_one)
    for c in chunks_one:
        print(f"[ONE Chunk Len: {len(c['content_with_weight'])}]")


if __name__ == "__main__":
    test_pipeline()
