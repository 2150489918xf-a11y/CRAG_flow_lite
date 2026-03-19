"""查看 ES 中 agent 知识库的分块内容"""
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
r = es.search(
    index="ragflow_lite_agent",
    body={
        "size": 10,
        "_source": ["content_with_weight", "docnm_kwd", "doc_type_kwd"],
        "query": {"match_all": {}},
    },
    request_timeout=10,
)

total = r["hits"]["total"]["value"]
hits = r["hits"]["hits"]

print(f"=== 总 chunks 数: {total} ===\n")

for i, h in enumerate(hits):
    src = h["_source"]
    docnm = src.get("docnm_kwd", "?")
    dtype = src.get("doc_type_kwd", "?")
    content = src.get("content_with_weight", "")
    chars = len(content)
    print(f"--- Chunk {i+1}/{total}  |  来源: {docnm}  |  类型: {dtype}  |  长度: {chars} 字符 ---")
    print(content[:500])
    if chars > 500:
        print(f"  ... (截断，完整 {chars} 字符)")
    print()
