"""
词权重计算 (精简自 RAGFlow rag/nlp/term_weight.py)
基于 IDF 和 NER 的词项权重计算
"""
import logging
import math
import json
import re
import os

import numpy as np
from rag.nlp import tokenizer as rag_tokenizer_module
from common.paths import NLP_RES_DIR

rag_tokenizer = rag_tokenizer_module


class Dealer:
    def __init__(self):
        self.stop_words = set([
            "请问", "您", "你", "我", "他", "是", "的", "就", "有", "于",
            "及", "即", "在", "为", "最", "从", "以", "了", "将", "与",
            "吗", "吧", "中", "#", "什么", "怎么", "哪个", "哪些", "啥", "相关",
        ])

        self.ne = {}
        self.df = {}

        try:
            ner_path = NLP_RES_DIR / "ner.json"
            if os.path.exists(ner_path):
                with open(ner_path, "r", encoding="utf-8") as f:
                    self.ne = json.load(f)
        except Exception:
            logging.warning("Load ner.json FAIL!")

    def pretoken(self, txt, num=False, stpwd=True):
        patt = [
            r'[~\u2014\t @#%!<>,\.?\\":\';{}\[\]_=()|,\u3002?\u300b\u2022\u25cf\u25cb\u2193\u300a;\u2018\u2019:\u201c\u201d\u3010\xa5 \u3011\u2026\uffe5!\u3001\u00b7()\u00d7`&\\/\u300c\u300d\\]'
        ]

        res = []
        for t in rag_tokenizer.tokenize(txt).split():
            tk = t
            if (stpwd and tk in self.stop_words) or (
                    re.match(r"[0-9]$", tk) and not num):
                continue
            skip = False
            for p in patt:
                if re.match(p, t):
                    skip = True
                    break
            if not skip and tk:
                res.append(tk)
        return res

    def token_merge(self, tks):
        def one_term(t):
            return len(t) == 1 or re.match(r"[0-9a-z]{1,2}$", t)

        res, i = [], 0
        while i < len(tks):
            j = i
            if i == 0 and one_term(tks[i]) and len(tks) > 1 and (
                    len(tks[i + 1]) > 1 and not re.match(r"[0-9a-zA-Z]", tks[i + 1])):
                res.append(" ".join(tks[0:2]))
                i = 2
                continue

            while j < len(tks) and tks[j] and tks[j] not in self.stop_words and one_term(tks[j]):
                j += 1
            if j - i > 1:
                if j - i < 5:
                    res.append(" ".join(tks[i:j]))
                    i = j
                else:
                    res.append(" ".join(tks[i:i + 2]))
                    i = i + 2
            else:
                if len(tks[i]) > 0:
                    res.append(tks[i])
                i += 1
        return [t for t in res if t]

    def split(self, txt):
        tks = []
        for t in re.sub(r"[ \t]+", " ", txt).split():
            if tks and re.match(r".*[a-zA-Z]$", tks[-1]) and \
                    re.match(r".*[a-zA-Z]$", t) and tks and \
                    self.ne.get(t, "") != "func" and self.ne.get(tks[-1], "") != "func":
                tks[-1] = tks[-1] + " " + t
            else:
                tks.append(t)
        return tks

    def weights(self, tks, preprocess=True):
        num_pattern = re.compile(r"[0-9,.]{2,}$")
        short_letter_pattern = re.compile(r"[a-z]{1,2}$")
        num_space_pattern = re.compile(r"[0-9. -]{2,}$")
        letter_pattern = re.compile(r"[a-z. -]+$")

        def ner(t):
            if num_pattern.match(t):
                return 2
            if short_letter_pattern.match(t):
                return 0.01
            if not self.ne or t not in self.ne:
                return 1
            m = {"toxic": 2, "func": 1, "corp": 3, "loca": 3, "sch": 3, "stock": 3, "firstnm": 1}
            return m.get(self.ne[t], 1)

        def postag(t):
            t = rag_tokenizer.tag(t)
            if t in set(["r", "c", "d"]):
                return 0.3
            if t in set(["ns", "nt"]):
                return 3
            if t in set(["n"]):
                return 2
            if re.match(r"[0-9-]+", t):
                return 2
            return 1

        def freq_fn(t):
            if num_space_pattern.match(t):
                return 3
            s = rag_tokenizer.freq(t)
            if not s and letter_pattern.match(t):
                return 300
            if not s:
                s = 0
            if not s and len(t) >= 4:
                s_tks = [tt for tt in rag_tokenizer.fine_grained_tokenize(t).split() if len(tt) > 1]
                if len(s_tks) > 1:
                    s = np.min([freq_fn(tt) for tt in s_tks]) / 6.
                else:
                    s = 0
            return max(s, 10)

        def idf(s, N):
            return math.log10(10 + ((N - s + 0.5) / (s + 0.5)))

        tw = []
        if not preprocess:
            idf1 = np.array([idf(freq_fn(t), 10000000) for t in tks])
            wts = idf1 * np.array([ner(t) * postag(t) for t in tks])
            tw = list(zip(tks, [float(s) for s in wts]))
        else:
            for tk in tks:
                tt = self.token_merge(self.pretoken(tk, True))
                if not tt:
                    continue
                idf1 = np.array([idf(freq_fn(t), 10000000) for t in tt])
                wts = idf1 * np.array([ner(t) * postag(t) for t in tt])
                tw.extend(zip(tt, [float(s) for s in wts]))

        S = np.sum([s for _, s in tw])
        if S == 0:
            return [(t, 1.0 / max(len(tw), 1)) for t, _ in tw]
        return [(t, s / S) for t, s in tw]
