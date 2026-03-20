"""
NLP 模块测试：分词器、词权重、同义词、查询扩展、merge 算法
"""
import pytest


class TestTokenizer:
    """分词器测试"""

    def test_chinese_tokenize(self):
        from rag.nlp.tokenizer import RagTokenizer
        tk = RagTokenizer()
        result = tk.tokenize("今天天气真好，我想去北京旅游")
        tokens = result.split()
        assert len(tokens) > 0

    def test_english_tokenize(self):
        from rag.nlp.tokenizer import RagTokenizer
        tk = RagTokenizer()
        result = tk.tokenize("The quick brown fox jumps over the lazy dog")
        tokens = result.split()
        assert len(tokens) > 0

    def test_fine_grained_tokenize(self):
        from rag.nlp.tokenizer import RagTokenizer
        tk = RagTokenizer()
        coarse = tk.tokenize("中华人民共和国")
        fine = tk.fine_grained_tokenize(coarse)
        assert len(fine) >= len(coarse)

    def test_freq_and_tag(self):
        from rag.nlp.tokenizer import RagTokenizer
        tk = RagTokenizer()
        freq = tk.freq("的")
        assert isinstance(freq, (int, float))
        tag = tk.tag("北京")
        assert isinstance(tag, str)

    def test_utility_functions(self):
        from rag.nlp.tokenizer import is_chinese, is_number, is_alphabet
        assert is_chinese("你好世界") is True
        assert is_chinese("hello") is False
        assert is_number("123.45") is True
        assert is_alphabet("abc") is True


class TestNLPUtils:
    """NLP 工具函数测试"""

    def test_find_codec_utf8(self):
        from rag.nlp import find_codec
        assert find_codec("你好世界".encode("utf-8")) == "utf-8"

    def test_find_codec_gbk(self):
        from rag.nlp import find_codec
        codec = find_codec("你好世界".encode("gbk"))
        assert codec is not None

    def test_num_tokens(self):
        from rag.nlp import num_tokens_from_string
        count = num_tokens_from_string("Hello world, 你好世界!")
        assert count > 0

    def test_naive_merge(self):
        from rag.nlp import naive_merge
        sections = [
            ("这是第一段文本。", "text"),
            ("这是第二段文本。", "text"),
            ("这是第三段比较长的文本，" * 50, "text"),
        ]
        chunks = naive_merge(sections, chunk_token_num=128)
        assert len(chunks) > 0

    def test_hierarchical_merge(self):
        from rag.nlp import hierarchical_merge
        sections = [
            ("这是一段关于人工智能的介绍。" * 10, "text"),
            ("机器学习是AI的核心技术。" * 10, "text"),
            ("深度学习推动了计算机视觉的发展。" * 10, "text"),
        ]
        # hierarchical_merge(bull, sections, depth)
        result = hierarchical_merge(0, sections, 2)
        assert isinstance(result, list)


class TestTermWeight:
    """词权重测试"""

    def test_weights(self):
        from rag.nlp.term_weight import Dealer
        tw = Dealer()
        weights = tw.weights(["北京", "旅游", "攻略"])
        assert len(weights) > 0
        for token, weight in weights:
            assert isinstance(weight, float)


class TestSynonym:
    """同义词测试"""

    def test_lookup(self):
        from rag.nlp.synonym import Dealer
        syn = Dealer()
        # 只要不抛异常即可，词典可能没有结果
        result = syn.lookup("happy")
        assert isinstance(result, list)


class TestQuery:
    """查询扩展测试"""

    def test_chinese_query(self):
        from rag.nlp.query import FulltextQueryer
        qryr = FulltextQueryer()
        match_expr, keywords = qryr.question("什么是人工智能？")
        # match_expr 可能为 None（输入太短）
        assert isinstance(keywords, list)

    def test_english_query(self):
        from rag.nlp.query import FulltextQueryer
        qryr = FulltextQueryer()
        match_expr, keywords = qryr.question("What is artificial intelligence?")
        assert isinstance(keywords, list)
