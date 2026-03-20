"""
rag_tokenizer 兼容层
为 DeepDoc 的 pdf_parser.py 提供 tokenize / tag / is_chinese 等接口
底层使用 jieba 分词（替代原版的 infinity.rag_tokenizer C++ 扩展）
"""
import re
import jieba
import jieba.posseg as pseg


def tokenize(text: str) -> str:
    """分词，返回空格分隔的 token 字符串"""
    if not text or not text.strip():
        return ""
    tokens = jieba.lcut(text.strip())
    return " ".join(tokens)


def fine_grained_tokenize(text: str) -> str:
    """细粒度分词"""
    if not text or not text.strip():
        return ""
    tokens = jieba.lcut_for_search(text.strip())
    return " ".join(tokens)


def tag(word: str) -> str:
    """词性标注，返回词性标签"""
    if not word:
        return ""
    pairs = pseg.lcut(word)
    if pairs:
        return pairs[0].flag
    return ""


def freq(word: str) -> int:
    """返回词频"""
    return jieba.dt.FREQ.get(word, 0)


def is_chinese(s: str) -> bool:
    """判断字符串是否包含中文"""
    if not s:
        return False
    for c in s:
        if '\u4e00' <= c <= '\u9fff':
            return True
    return False


def is_number(s: str) -> bool:
    """判断是否为数字"""
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return bool(re.match(r'^[0-9,.%%]+$', s))


def is_alphabet(s: str) -> bool:
    """判断是否为字母"""
    if not s:
        return False
    return bool(re.match(r'^[a-zA-Z]+$', s))


def naive_qie(txt: str) -> str:
    """简单切分"""
    return tokenize(txt)


# 兼容原版接口：模块级函数 + tokenizer 对象
class _RagTokenizer:
    def tokenize(self, text):
        return tokenize(text)

    def fine_grained_tokenize(self, text):
        return fine_grained_tokenize(text)

    def tag(self, word):
        return tag(word)

    def freq(self, word):
        return freq(word)

    @staticmethod
    def _tradi2simp(text):
        return text  # 简体繁体转换暂不实现

    @staticmethod
    def _strQ2B(text):
        """全角转半角"""
        result = []
        for c in text:
            code = ord(c)
            if code == 0x3000:
                code = 0x0020
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            result.append(chr(code))
        return ''.join(result)


tokenizer = _RagTokenizer()
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B
