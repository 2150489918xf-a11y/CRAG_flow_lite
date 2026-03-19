"""
基于 jieba 的中英文分词器
替代 RAGFlow 的 infinity.rag_tokenizer，提供等价接口
"""
import re
import logging

import jieba
import jieba.posseg as pseg

# 抑制 jieba 加载日志
jieba.setLogLevel(logging.WARNING)


class RagTokenizer:
    """
    RAGFlow Lite 分词器 (jieba 实现)

    提供与 RAGFlow rag_tokenizer 兼容的接口:
    - tokenize(text) -> 空格分隔的 token 字符串
    - fine_grained_tokenize(coarse_tokens) -> 更细粒度的分词
    - freq(token) -> 词频
    - tag(token) -> 词性
    """

    def __init__(self):
        self._freq_cache = {}

    @staticmethod
    def _is_chinese_char(ch):
        cp = ord(ch)
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F or
                0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or
                0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)

    @staticmethod
    def _strQ2B(text):
        """全角转半角"""
        res = []
        for ch in text:
            code = ord(ch)
            if code == 0x3000:
                code = 0x0020
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            res.append(chr(code))
        return "".join(res)

    @staticmethod
    def _tradi2simp(text):
        """繁体转简体 (简单实现，不依赖外部库)"""
        try:
            import opencc
            converter = opencc.OpenCC('t2s')
            return converter.convert(text)
        except ImportError:
            return text

    def tokenize(self, text):
        """
        粗粒度分词，返回空格分隔的 token 字符串
        与 RAGFlow rag_tokenizer.tokenize() 等价
        """
        if not text or not text.strip():
            return ""

        # 预处理：全角转半角，转小写
        text = self._strQ2B(text).lower()
        # 去除多余空白
        text = re.sub(r"\s+", " ", text).strip()

        # jieba 分词
        tokens = jieba.cut(text, cut_all=False)
        # 过滤空白 token
        result = [t.strip() for t in tokens if t.strip()]
        return " ".join(result)

    def fine_grained_tokenize(self, coarse_tokens):
        """
        细粒度分词：对粗粒度分词结果中的长词再做切分
        输入是空格分隔的 token 字符串，输出也是
        """
        if not coarse_tokens:
            return ""

        tokens = coarse_tokens.split()
        result = []
        for token in tokens:
            # 短 token 或纯英文/数字不再切分
            if len(token) <= 2 or re.match(r"^[a-z0-9\.\+#_\*\-]+$", token):
                result.append(token)
                continue

            # 对长中文词使用 jieba 搜索模式再切分
            has_chinese = any(self._is_chinese_char(ch) for ch in token)
            if has_chinese and len(token) >= 3:
                sub_tokens = list(jieba.cut_for_search(token))
                sub_tokens = [t.strip() for t in sub_tokens if t.strip()]
                if len(sub_tokens) > 1:
                    result.extend(sub_tokens)
                else:
                    result.append(token)
            else:
                result.append(token)

        return " ".join(result)

    def freq(self, token):
        """返回词频 (基于 jieba 内部词典)"""
        if not token:
            return 0
        return jieba.get_FREQ(token) or 0

    def tag(self, token):
        """返回词性标注"""
        if not token:
            return ""
        pairs = list(pseg.cut(token))
        if pairs:
            return pairs[0].flag
        return ""


def is_chinese(s):
    """判断是否包含中文"""
    if not s:
        return False
    chinese = sum(1 for ch in s if '\u4e00' <= ch <= '\u9fff')
    return (chinese / max(len(s), 1)) > 0.2


def is_number(s):
    """判断是否为数字"""
    if not s:
        return False
    return bool(re.match(r'^[0-9]+\.?[0-9]*$', s))


def is_alphabet(s):
    """判断是否为纯字母"""
    if not s:
        return False
    return bool(re.match(r'^[a-zA-Z]+$', s))


# 模块级单例
tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B
