"""
查询扩展 (精简自 RAGFlow rag/nlp/query.py)
构建 ES query_string 查询，包含分词加权、同义词扩展、混合相似度计算
"""
import logging
import json
import re
from collections import defaultdict

from rag.nlp import tokenizer as rag_tokenizer_module
from rag.nlp import term_weight, synonym

rag_tokenizer = rag_tokenizer_module


class MatchTextExpr:
    """全文检索表达式"""
    def __init__(self, fields, matching_text, topn=100, extra_options=None):
        self.fields = fields
        self.matching_text = matching_text
        self.topn = topn
        self.extra_options = extra_options or {}


class MatchDenseExpr:
    """向量检索表达式"""
    def __init__(self, vector_column_name, embedding_data, dtype, metric, topn, extra_options=None):
        self.vector_column_name = vector_column_name
        self.embedding_data = embedding_data
        self.dtype = dtype
        self.metric = metric
        self.topn = topn
        self.extra_options = extra_options or {}


class FusionExpr:
    """融合表达式"""
    def __init__(self, method, topn, fusion_params=None):
        self.method = method
        self.topn = topn
        self.fusion_params = fusion_params or {}


class FulltextQueryer:
    """
    全文检索 Query 构建器
    照搬 RAGFlow query.py 的核心逻辑
    """
    def __init__(self):
        self.tw = term_weight.Dealer()
        self.syn = synonym.Dealer()
        self.query_fields = [
            "title_tks^10",
            "title_sm_tks^5",
            "important_kwd^30",
            "important_tks^20",
            "question_tks^20",
            "content_ltks^2",
            "content_sm_ltks",
        ]

    @staticmethod
    def is_chinese(txt):
        if not txt:
            return False
        chinese = sum(1 for ch in txt if '\u4e00' <= ch <= '\u9fff')
        return chinese / max(len(txt), 1) > 0.2

    @staticmethod
    def add_space_between_eng_zh(txt):
        return re.sub(r"([a-zA-Z])([一-鿿])", r"\1 \2",
                      re.sub(r"([一-鿿])([a-zA-Z])", r"\1 \2", txt))

    @staticmethod
    def rmWWW(txt):
        return re.sub(
            r"(?:https?://|www\.)[a-zA-Z0-9./?#&=_%+-]+",
            "",
            txt,
        ).strip()

    @staticmethod
    def sub_special_char(tk):
        return re.sub(r"([+\-!(){}\[\]^\"~*?:/])", r"\\\1", tk)

    def question(self, txt, tbl="qa", min_match=0.6):
        """
        构建 ES query_string 查询
        返回 (MatchTextExpr, keywords_list)
        """
        original_query = txt
        txt = self.add_space_between_eng_zh(txt)
        txt = re.sub(
            r"[ :|\r\n\t,，。？?/`!！&^%%()[\]{}<>]+",
            " ",
            rag_tokenizer.strQ2B(txt.lower()),
        ).strip()
        txt = self.rmWWW(txt)

        if not self.is_chinese(txt):
            # 英文查询
            tks = rag_tokenizer.tokenize(txt).split()
            keywords = [t for t in tks if t]
            tks_w = self.tw.weights(tks, preprocess=False)
            tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w]
            tks_w = [(re.sub(r"^[+-]", "", tk), w) for tk, w in tks_w if tk]
            tks_w = [(tk.strip(), w) for tk, w in tks_w if tk.strip()]

            syns = []
            for tk, w in tks_w[:256]:
                syn = [rag_tokenizer.tokenize(s) for s in self.syn.lookup(tk)]
                keywords.extend(syn)
                syn = ["\"{}\"^{:.4f}".format(s, w / 4.) for s in syn if s.strip()]
                syns.append(" ".join(syn))

            q = [
                "({}^{:.4f}".format(tk, w) + " {})".format(syn)
                for (tk, w), syn in zip(tks_w, syns)
                if tk and not re.match(r"[.^+()-]", tk)
            ]
            for i in range(1, len(tks_w)):
                left, right = tks_w[i - 1][0].strip(), tks_w[i][0].strip()
                if not left or not right:
                    continue
                q.append(
                    '"%s %s"^%.4f' % (left, right, max(tks_w[i - 1][1], tks_w[i][1]) * 2)
                )
            if not q:
                q.append(txt)
            query = " ".join(q)
            return MatchTextExpr(
                self.query_fields, query, 100, {"original_query": original_query}
            ), keywords

        # 中文查询
        def need_fine_grained_tokenize(tk):
            if len(tk) < 3:
                return False
            if re.match(r"[0-9a-z.+#_*-]+$", tk):
                return False
            return True

        qs, keywords = [], []
        for tt in self.tw.split(txt)[:256]:
            if not tt:
                continue
            keywords.append(tt)
            twts = self.tw.weights([tt])
            syns = self.syn.lookup(tt)
            if syns and len(keywords) < 32:
                keywords.extend(syns)

            tms = []
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                sm = (
                    rag_tokenizer.fine_grained_tokenize(tk).split()
                    if need_fine_grained_tokenize(tk)
                    else []
                )
                sm = [
                    re.sub(
                        r"[ ,./;'\[\]\\`~!@#$%^&*()=+_<>?:\"\{\}|，。；''【】、！￥……（）——《》？：""-]+",
                        "", m,
                    )
                    for m in sm
                ]
                sm = [self.sub_special_char(m) for m in sm if len(m) > 1]
                sm = [m for m in sm if len(m) > 1]

                if len(keywords) < 32:
                    keywords.append(re.sub(r"[ \\\"']+", "", tk))
                    keywords.extend(sm)

                tk_syns = self.syn.lookup(tk)
                tk_syns = [self.sub_special_char(s) for s in tk_syns]
                if len(keywords) < 32:
                    keywords.extend([s for s in tk_syns if s])
                tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
                tk_syns = [f'"{s}"' if s.find(" ") > 0 else s for s in tk_syns]

                if len(keywords) >= 32:
                    break

                tk = self.sub_special_char(tk)
                if tk.find(" ") > 0:
                    tk = '"%s"' % tk
                if tk_syns:
                    tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
                if sm:
                    tk = f'{tk} OR "%s" OR ("%s"~2)^0.5' % (" ".join(sm), " ".join(sm))
                if tk.strip():
                    tms.append((tk, w))

            tms_str = " ".join([f"({t})^{w}" for t, w in tms])

            if len(twts) > 1:
                tms_str += ' ("%s"~2)^1.5' % rag_tokenizer.tokenize(tt)

            syns_str = " OR ".join(
                ['"%s"' % rag_tokenizer.tokenize(self.sub_special_char(s)) for s in syns]
            )
            if syns_str and tms_str:
                tms_str = f"({tms_str})^5 OR ({syns_str})^0.7"

            qs.append(tms_str)

        if qs:
            query = " OR ".join([f"({t})" for t in qs if t])
            if not query:
                query = txt
            return MatchTextExpr(
                self.query_fields, query, 100,
                {"minimum_should_match": min_match, "original_query": original_query}
            ), keywords
        return None, keywords

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
        """混合相似度：向量余弦 + token 相似度的加权融合"""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        sims = cosine_similarity([avec], bvecs)
        tksim = self.token_similarity(atks, btkss)
        if np.sum(sims[0]) == 0:
            return np.array(tksim), tksim, sims[0]
        return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        """Token 级别相似度"""
        def to_dict(tks):
            if isinstance(tks, str):
                tks = tks.split()
            d = defaultdict(int)
            wts = self.tw.weights(tks, preprocess=False)
            for i, (t, c) in enumerate(wts):
                d[t] += c * 0.4
                if i + 1 < len(wts):
                    _t, _c = wts[i + 1]
                    d[t + _t] += max(c, _c) * 0.6
            return d

        atks = to_dict(atks)
        btkss = [to_dict(tks) for tks in btkss]
        return [self._sim(atks, btks) for btks in btkss]

    @staticmethod
    def _sim(qtwt, dtwt):
        """词项加权相似度计算"""
        s = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                s += v
        q = 1e-9
        for k, v in qtwt.items():
            q += v
        return s / q
