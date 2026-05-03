"""Cross-encoder reranker.

Cross-encoders score (query, document) pairs jointly — far more accurate
than bi-encoder cosine but ~10x slower per pair. We only run them on the
top-N RRF candidates and keep the top-K after reranking.

Dependency: `sentence-transformers` (pulls in PyTorch). It's heavy. If it
isn't installed the reranker degrades to a no-op and the retriever returns
the unmodified RRF list — the rest of the system continues to work.
"""
from __future__ import annotations

import threading
from typing import Sequence

from src.config import RERANKER_MODEL


_LOCK = threading.Lock()
_MODEL = None
_LOAD_FAILED = False


def is_available() -> bool:
    """True if sentence-transformers can be imported AND the model loads."""
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None:
        return True
    if _LOAD_FAILED:
        return False
    with _LOCK:
        if _MODEL is not None:
            return True
        if _LOAD_FAILED:
            return False
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception:
            _LOAD_FAILED = True
            return False
        try:
            _MODEL = CrossEncoder(RERANKER_MODEL, max_length=512)
            return True
        except Exception:
            _LOAD_FAILED = True
            return False


def rerank(query: str, candidates: Sequence[dict], top_k: int) -> list[dict]:
    """Score (query, candidate.text) pairs and return the top-K candidates.

    Each candidate is a dict with at least a 'text' field; the returned
    list is the same shape with an added 'rerank_score' field.
    """
    if not candidates:
        return []
    if not is_available():
        return list(candidates)[:top_k]
    pairs = [(query, c["text"]) for c in candidates]
    try:
        scores = _MODEL.predict(pairs).tolist()  # type: ignore[union-attr]
    except Exception:
        return list(candidates)[:top_k]
    indexed = list(zip(candidates, scores))
    indexed.sort(key=lambda x: -x[1])
    out = []
    for c, s in indexed[:top_k]:
        c2 = dict(c)
        c2["rerank_score"] = float(s)
        out.append(c2)
    return out
