import os
import sys

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.app.chunking import chunk
import json

def test_deepdoc_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), "test_doc.pdf")

    print(f"Target PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"Warning: {pdf_path} not found.")
        return

    configs = [
        {"parser_id": "naive", "use_parent_child": True},
        {"parser_id": "qa"},
        {"parser_id": "laws"},
        {"parser_id": "one"}
    ]

    for cfg in configs:
        pid = cfg["parser_id"]
        print(f"\n======================================")
        print(f"Testing DeepDoc parsing with STRATEGY: {pid.upper()}")
        print(f"======================================")
        
        try:
            chunks = chunk(pdf_path, parser_config=cfg)
            print(f"=> Successfully generated {len(chunks)} chunks using strategy {pid}!")
            
            # Print a few to check
            for i, ck in enumerate(chunks[:3]):
                print(f"\n--- Chunk {i+1} ---")
                print(f"ID: {ck.get('id')}")
                print(f"Type: {ck.get('chunk_type_kwd')} | DocType: {ck.get('doc_type_kwd')}")
                if "parent_id_kwd" in ck:
                    print(f"Parent_id: {ck.get('parent_id_kwd')}")
                content = ck.get('content_with_weight', '')
                print(f"Content length: {len(content)}")
                print(f"Preview:\n{content[:150]}...")
        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_deepdoc_pdf()
