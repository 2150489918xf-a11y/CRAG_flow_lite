"""
同义词查找 (精简自 RAGFlow rag/nlp/synonym.py)
"""
import logging
import json
import os
import re

from rag.settings import get_project_base_directory


class Dealer:
    def __init__(self):
        self.dictionary = {}
        path = os.path.join(get_project_base_directory(), "res", "synonym.json")
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.dictionary = json.load(f)
                self.dictionary = {
                    (k.lower() if isinstance(k, str) else k): v
                    for k, v in self.dictionary.items()
                }
            else:
                logging.warning(f"synonym.json not found at {path}")
        except Exception:
            logging.warning("Failed to load synonym.json")

        if not self.dictionary:
            logging.warning("Synonym dictionary is empty")

    def lookup(self, tk, topn=8):
        if not tk or not isinstance(tk, str):
            return []

        key = re.sub(r"[ \t]+", " ", tk.strip()).lower()
        res = self.dictionary.get(key, [])
        if isinstance(res, str):
            res = [res]
        if res:
            return res[:topn]

        # 纯英文 fallback 到 WordNet
        if re.fullmatch(r"[a-z]+", tk):
            try:
                from nltk.corpus import wordnet
                wn_set = {
                    re.sub("_", " ", syn.name().split(".")[0])
                    for syn in wordnet.synsets(tk)
                }
                wn_set.discard(tk)
                return [t for t in wn_set if t][:topn]
            except Exception:
                pass

        return []
