"""Hand-rolled BM25 over the corpus. Used as the lexical leg of a hybrid retriever.

Reference: Robertson & Zaragoza (2009). k1=1.5, b=0.75 are standard defaults.
"""
from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+", re.UNICODE)
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are",
    "was", "were", "be", "been", "by", "for", "with", "as", "that", "this",
    "it", "from", "his", "her", "its", "their", "they", "he", "she",
}


def _tok(s: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(s.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self._tokens = [_tok(d["text"]) for d in docs]
        self._lengths = [len(t) for t in self._tokens] or [0]
        self._avglen = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        df: Counter = Counter()
        for toks in self._tokens:
            for term in set(toks):
                df[term] += 1
        N = max(1, len(self._tokens))
        self._idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self._tf = [Counter(toks) for toks in self._tokens]

    def score_all(self, query: str) -> list[float]:
        q_terms = _tok(query)
        if not q_terms or not self.docs:
            return [0.0] * len(self.docs)
        scores = [0.0] * len(self.docs)
        for i, tf in enumerate(self._tf):
            dl = self._lengths[i] or 1
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avglen or 1))
                s += idf * (f * (self.k1 + 1)) / denom
            scores[i] = s
        return scores

    def top_k(self, query: str, k: int) -> list[tuple[dict, float]]:
        scores = self.score_all(query)
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.docs[i], scores[i]) for i in idx if scores[i] > 0]
