"""
RAGFlow Lite 核心模块测试
测试分词器、NLP 工具、分块引擎、文档解析器、配置加载
"""
import os
import sys
import json

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config():
    """测试配置加载"""
    print("=== 测试配置加载 ===")
    from rag.settings import get_config, get_es_config, get_embedding_config, get_rag_config

    config = get_config()
    assert config is not None, "配置加载失败"
    assert "es" in config, "缺少 es 配置"
    assert "embedding" in config, "缺少 embedding 配置"

    es_cfg = get_es_config()
    assert "hosts" in es_cfg, "缺少 ES hosts"

    emb_cfg = get_embedding_config()
    assert "model_name" in emb_cfg, "缺少 embedding model_name"

    rag_cfg = get_rag_config()
    assert rag_cfg["chunk_token_num"] == 512

    print(f"  ✅ 配置加载正常: ES={es_cfg['hosts']}, Model={emb_cfg['model_name']}")
    print(f"  ✅ RAG 参数: chunk_token_num={rag_cfg['chunk_token_num']}")


def test_tokenizer():
    """测试分词器"""
    print("\n=== 测试分词器 ===")
    from rag.nlp.tokenizer import RagTokenizer, is_chinese, is_number, is_alphabet

    tk = RagTokenizer()

    # 中文分词
    result = tk.tokenize("今天天气真好，我想去北京旅游")
    tokens = result.split()
    assert len(tokens) > 0, "中文分词结果为空"
    print(f"  ✅ 中文分词: '{result}' ({len(tokens)} tokens)")

    # 英文分词
    result_en = tk.tokenize("The quick brown fox jumps over the lazy dog")
    tokens_en = result_en.split()
    assert len(tokens_en) > 0, "英文分词结果为空"
    print(f"  ✅ 英文分词: '{result_en}' ({len(tokens_en)} tokens)")

    # 细粒度分词
    fine = tk.fine_grained_tokenize(tk.tokenize("中华人民共和国"))
    print(f"  ✅ 细粒度分词: '{fine}'")

    # 词频
    freq = tk.freq("的")
    print(f"  ✅ 词频('的'): {freq}")

    # 词性
    tag = tk.tag("北京")
    print(f"  ✅ 词性('北京'): {tag}")

    # 工具函数
    assert is_chinese("你好世界") == True
    assert is_chinese("hello") == False
    assert is_number("123.45") == True
    assert is_alphabet("abc") == True
    print(f"  ✅ 工具函数正常")


def test_nlp_utils():
    """测试 NLP 工具函数"""
    print("\n=== 测试 NLP 工具函数 ===")
    from rag.nlp import find_codec, num_tokens_from_string, naive_merge

    # 编码检测
    text = "你好世界"
    codec = find_codec(text.encode("utf-8"))
    assert codec == "utf-8", f"编码检测失败: {codec}"
    print(f"  ✅ 编码检测: utf-8")

    codec_gbk = find_codec(text.encode("gbk"))
    print(f"  ✅ 编码检测(gbk): {codec_gbk}")

    # Token 计数
    count = num_tokens_from_string("Hello world, 你好世界!")
    assert count > 0, "token 计数为 0"
    print(f"  ✅ Token 计数: {count}")

    # naive_merge 分块
    sections = [
        ("这是第一段文本。", "text"),
        ("这是第二段文本。", "text"),
        ("这是第三段比较长的文本，" * 50, "text"),
    ]
    chunks = naive_merge(sections, chunk_token_num=128)
    assert len(chunks) > 0, "分块结果为空"
    print(f"  ✅ naive_merge: {len(sections)} 段 → {len(chunks)} 块")
    for i, ck in enumerate(chunks):
        tk_count = num_tokens_from_string(ck)
        print(f"    chunk[{i}]: {tk_count} tokens, {len(ck)} chars")


def test_term_weight():
    """测试词权重"""
    print("\n=== 测试词权重 ===")
    from rag.nlp.term_weight import Dealer

    tw = Dealer()
    tokens = ["北京", "旅游", "攻略"]
    weights = tw.weights(tokens)
    assert len(weights) > 0, "词权重结果为空"
    print(f"  ✅ 词权重计算:")
    for t, w in weights:
        print(f"    {t}: {w:.4f}")


def test_synonym():
    """测试同义词"""
    print("\n=== 测试同义词 ===")
    from rag.nlp.synonym import Dealer

    syn = Dealer()
    # 测试同义词查找
    result = syn.lookup("happy")
    print(f"  ✅ 同义词('happy'): {result[:5] if result else '(词典中无)'}")

    result2 = syn.lookup("电脑")
    print(f"  ✅ 同义词('电脑'): {result2[:5] if result2 else '(词典中无)'}")


def test_query():
    """测试查询扩展"""
    print("\n=== 测试查询扩展 ===")
    from rag.nlp.query import FulltextQueryer

    qryr = FulltextQueryer()

    # 中文查询
    match_expr, keywords = qryr.question("什么是人工智能？")
    if match_expr:
        print(f"  ✅ 中文查询构建成功")
        print(f"    query_string: {match_expr.matching_text[:100]}...")
        print(f"    keywords: {keywords[:10]}")
    else:
        print(f"  ⚠️ 中文查询构建返回 None (可能输入太短)")

    # 英文查询
    match_expr_en, keywords_en = qryr.question("What is artificial intelligence?")
    if match_expr_en:
        print(f"  ✅ 英文查询构建成功")
        print(f"    keywords: {keywords_en[:10]}")


def test_parsers():
    """测试文档解析器"""
    print("\n=== 测试文档解析器 ===")

    # TXT 解析
    from rag.parser.other_parsers import parse_txt
    txt_content = "第一段落\n\n第二段落\n\n第三段落"
    sections, tables = parse_txt("test.txt", txt_content.encode("utf-8"))
    assert len(sections) == 3, f"TXT 解析段落数错误: {len(sections)}"
    print(f"  ✅ TXT 解析: {len(sections)} 段")

    # JSON 解析
    from rag.parser.other_parsers import parse_json
    json_data = json.dumps({"name": "RAGFlow", "version": "1.0", "features": ["search", "embed"]})
    sections_json, _ = parse_json("test.json", json_data.encode("utf-8"))
    assert len(sections_json) > 0
    print(f"  ✅ JSON 解析: {len(sections_json)} 条")
    for s in sections_json[:3]:
        print(f"    {s[0][:60]}")

    # Markdown 解析
    from rag.parser.markdown_parser import parse
    md_content = "# 标题一\n内容一\n## 标题二\n内容二\n### 标题三\n内容三"
    sections_md, _ = parse("test.md", md_content.encode("utf-8"))
    assert len(sections_md) > 0
    print(f"  ✅ Markdown 解析: {len(sections_md)} 段")

    # HTML 解析
    from rag.parser.other_parsers import parse_html
    html_content = "<html><body><h1>Title</h1><p>Content paragraph</p><table><tr><td>A</td></tr></table></body></html>"
    sections_html, tables_html = parse_html("test.html", html_content.encode("utf-8"))
    print(f"  ✅ HTML 解析: {len(sections_html)} 段, {len(tables_html)} 表")


def test_chunking():
    """测试完整分块流程"""
    print("\n=== 测试完整分块流程 ===")
    from rag.app.chunking import chunk

    # 创建测试文本
    test_text = """
    RAGFlow 是一个基于深度文档理解的开源 RAG 引擎。
    它提供了简洁的 RAG 工作流，适用于任何规模的企业。

    核心特性包括：
    1. 深度文档理解 — 支持多种文档格式
    2. 混合检索 — 全文检索 + 向量检索
    3. 智能分块 — 基于语义的文档分块
    4. 查询扩展 — 同义词 + 关键词加权

    RAGFlow Lite 是其轻量版本，适合比赛场景。
    """

    # 写临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(test_text)
        tmp_path = f.name

    try:
        chunks = chunk(tmp_path, lang="Chinese")
        assert len(chunks) > 0, "分块结果为空"
        print(f"  ✅ TXT 分块: {len(chunks)} 个 chunks")
        for i, ck in enumerate(chunks):
            content = ck.get("content_with_weight", "")
            ltks = ck.get("content_ltks", "")
            sm_ltks = ck.get("content_sm_ltks", "")
            print(f"    chunk[{i}]: {len(content)} chars, "
                  f"{len(ltks.split())} coarse_tks, "
                  f"{len(sm_ltks.split())} fine_tks")
            assert "id" in ck, "chunk 缺少 id"
            assert "docnm_kwd" in ck, "chunk 缺少 docnm_kwd"
            assert "content_ltks" in ck, "chunk 缺少 content_ltks"
    finally:
        os.unlink(tmp_path)


def test_graph_extractor_dataclasses():
    """测试 GraphRAG 数据结构"""
    print("\n=== 测试 GraphRAG 数据结构 ===")
    from rag.graph.extractor import Entity, Relation, ExtractionResult

    e = Entity(name="微软", type="ORGANIZATION", description="科技公司")
    assert e.name == "微软"
    print(f"  ✅ Entity: {e.name} ({e.type})")

    r = Relation(source="微软", target="OpenAI", description="投资")
    assert r.source == "微软"
    print(f"  ✅ Relation: {r.source} → {r.target}")

    result = ExtractionResult(entities=[e], relations=[r])
    assert len(result.entities) == 1
    assert len(result.relations) == 1
    print(f"  ✅ ExtractionResult: {len(result.entities)} entities, {len(result.relations)} relations")


def test_graph_store():
    """测试图谱存储 (NetworkX 部分，不需要 ES)"""
    print("\n=== 测试 NetworkX 图 + PageRank ===")
    import networkx as nx
    from rag.graph.extractor import Entity, Relation, ExtractionResult
    from rag.graph.graph_store import GraphStore

    # 构建模拟数据
    entities = [
        Entity(name="微软", type="ORGANIZATION", description="全球最大科技公司"),
        Entity(name="OpenAI", type="ORGANIZATION", description="AI 研究公司"),
        Entity(name="ChatGPT", type="PRODUCT", description="大语言模型"),
        Entity(name="Satya Nadella", type="PERSON", description="微软CEO"),
        Entity(name="Sam Altman", type="PERSON", description="OpenAI CEO"),
    ]
    relations = [
        Relation(source="微软", target="OpenAI", description="投资了130亿美元"),
        Relation(source="OpenAI", target="ChatGPT", description="开发了"),
        Relation(source="Satya Nadella", target="微软", description="是CEO"),
        Relation(source="Sam Altman", target="OpenAI", description="是CEO"),
    ]
    extraction = ExtractionResult(entities=entities, relations=relations)

    # 构建图 (不连 ES)
    store = GraphStore.__new__(GraphStore)
    store.graph = nx.DiGraph()
    store._entity_map = {}
    store.es_conn = None
    store.emb_mdl = None

    store.build_graph(extraction)
    assert store.graph.number_of_nodes() == 5, f"节点数错误: {store.graph.number_of_nodes()}"
    assert store.graph.number_of_edges() == 4, f"边数错误: {store.graph.number_of_edges()}"
    print(f"  ✅ 图构建: {store.graph.number_of_nodes()} 节点, {store.graph.number_of_edges()} 边")

    # PageRank
    pr = store.compute_pagerank()
    assert len(pr) == 5
    print(f"  ✅ PageRank 计算:")
    for node, score in sorted(pr.items(), key=lambda x: x[1], reverse=True):
        name = store.graph.nodes[node].get("name", node)
        print(f"    {name}: {score:.4f}")

    # N跳遍历
    neighbors = store.get_neighbors("ChatGPT", n_hops=2)
    print(f"  ✅ N跳遍历(ChatGPT, 2跳): {len(neighbors)} 个邻居")
    for nb in neighbors:
        print(f"    → {nb['name']} (关系: {nb['relation']}, 跳数: {nb['hop']}, PR: {nb['pagerank']:.3f})")

    # 保存/加载
    import tempfile
    tmp = tempfile.mktemp(suffix=".json")
    store.save_graph(tmp)
    assert os.path.exists(tmp)

    store2 = GraphStore.__new__(GraphStore)
    store2.graph = nx.DiGraph()
    store2._entity_map = {}
    store2.es_conn = None
    store2.emb_mdl = None
    store2.load_graph(tmp)
    assert store2.graph.number_of_nodes() == 5
    print(f"  ✅ 图持久化: 保存/加载成功")
    os.unlink(tmp)


def test_graph_search_format():
    """测试图谱检索结果格式化"""
    print("\n=== 测试图谱结果格式化 ===")
    from rag.graph.graph_search import GraphSearcher

    entities = [
        {"name": "微软", "type": "ORGANIZATION", "description": "科技公司", "pagerank": 0.85, "fusion_score": 0.9},
        {"name": "OpenAI", "type": "ORGANIZATION", "description": "AI研究", "pagerank": 0.72, "fusion_score": 0.8},
    ]
    relations = [
        {"source": "微软", "target": "OpenAI", "description": "微软 投资了130亿美元 OpenAI", "pagerank": 0.85, "fusion_score": 0.9},
    ]
    paths = [
        {"from": "ChatGPT", "to": "OpenAI", "type": "ORGANIZATION", "relation": "开发了", "hop": 1, "pagerank": 0.72},
        {"from": "OpenAI", "to": "微软", "type": "ORGANIZATION", "relation": "投资了130亿美元", "hop": 2, "pagerank": 0.85},
    ]

    context = GraphSearcher.format_context(entities, relations, paths)
    assert "知识图谱上下文" in context
    assert "微软" in context
    assert "OpenAI" in context
    print(f"  ✅ 格式化输出 ({len(context)} chars):")
    for line in context.split("\n"):
        print(f"    {line}")


if __name__ == "__main__":
    print("=" * 60)
    print("RAGFlow Lite 核心模块测试")
    print("=" * 60)

    tests = [
        test_config,
        test_tokenizer,
        test_nlp_utils,
        test_term_weight,
        test_synonym,
        test_query,
        test_parsers,
        test_chunking,
        test_graph_extractor_dataclasses,
        test_graph_store,
        test_graph_search_format,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  ❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败 / {len(tests)} 总计")
    print("=" * 60)
