import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.app import chunking
import json

def test_phase2():
    # Create a dummy txt document mimicking a book, paper, manual, and PPT
    dummy_text = """
title@@[1.0, 2.0]这里是一篇伟大的学术论文标题
author@@[3.0, 4.0]张三，李四
abstract@@[5.0, 6.0]本论文提出了一种全新的 RAG 分块策略，大幅提升了召回率。
text@@[7.0, 8.0]在传统的检索增强生成中，遇到长文本常常会丢失上下文。
reference@@[9.0, 10.0][1] 张三. 2024. RAG 系统综述.

==page_number: 1==
第一章 绪论
1.1 研究背景
这是1.1节的内容。
1.1.1 进一步的背景
这是更深一层的内容。

==page_number: 2==
这是一份极其重要的第二页幻灯片文本。
绝不能和第一页混在一起。

"""
    
    with open("dummy_phase2.txt", "w", encoding="utf-8") as f:
        f.write(dummy_text)
        
    configs = ["book", "paper", "presentation", "table", "manual"]
    
    for pid in configs:
        print(f"\n======================================")
        print(f"Testing STRATEGY: {pid.upper()}")
        print(f"======================================")
        
        cfg = {"parser_id": pid}
        try:
            chunks = chunking.chunk("dummy_phase2.txt", lang="Chinese", parser_config=cfg)
            print(f"=> Successfully generated {len(chunks)} chunks!")
            for i, ck in enumerate(chunks):
                content = ck.get('content_with_weight', '').replace('\n', ' ')
                print(f"[{i+1}] {content[:80]}...")
        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_phase2()
