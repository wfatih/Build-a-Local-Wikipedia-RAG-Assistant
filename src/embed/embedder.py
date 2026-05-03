"""Embedding via local Ollama (`nomic-embed-text`).

Uses a persistent HTTP session (keep-alive) and the batch `/api/embed`
endpoint when available, with automatic fallback to the per-item
`/api/embeddings` endpoint on older Ollama builds. The output of both paths
is identical (same model, same dimensions, L2-normalised).

`nomic-embed-text` is a *task-conditioned* model: it expects each text to
be tagged with `search_query: ` or `search_document: `. Using the right
prefix gives a noticeable retrieval-quality lift; mixing them or omitting
them collapses queries and documents into the same untyped subspace.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import requests

from src.config import EMBED_DIM, EMBED_DOC_PREFIX, EMBED_MODEL, EMBED_QUERY_PREFIX, OLLAMA_HOST


_SESSION = requests.Session()
_BATCH_ENDPOINT_OK: bool | None = None  # tri-state: None=untested, True/False after first call
EMBED_BATCH_SIZE = 32


def _post(path: str, payload: dict, timeout: int = 600) -> requests.Response:
    return _SESSION.post(f"{OLLAMA_HOST}{path}", json=payload, timeout=timeout)


def _embed_one_legacy(text: str) -> np.ndarray:
    r = _post("/api/embeddings", {"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    vec = r.json().get("embedding")
    if not vec:
        raise RuntimeError("Ollama returned empty embedding (legacy endpoint)")
    return np.asarray(vec, dtype=np.float32)


def _embed_batch_new(texts: list[str]) -> np.ndarray:
    """Use Ollama's batch endpoint. Raises on HTTP/parse failure so callers
    can fall back to the legacy endpoint."""
    r = _post("/api/embed", {"model": EMBED_MODEL, "input": texts})
    r.raise_for_status()
    data = r.json()
    embs = data.get("embeddings")
    if not embs or len(embs) != len(texts):
        raise RuntimeError("Ollama batch endpoint returned an unexpected payload")
    return np.asarray(embs, dtype=np.float32)


def _embed_chunk(texts: list[str], retries: int = 3, backoff: float = 1.5) -> np.ndarray:
    """Embed a chunk of texts. Try batch endpoint, fall back per-item."""
    global _BATCH_ENDPOINT_OK
    last_exc: Exception | None = None

    for attempt in range(retries):
        try:
            if _BATCH_ENDPOINT_OK is not False:
                try:
                    out = _embed_batch_new(texts)
                    _BATCH_ENDPOINT_OK = True
                    return out
                except (requests.HTTPError, RuntimeError) as e:
                    if _BATCH_ENDPOINT_OK is None:
                        # First time: assume the build is too old, switch off batch.
                        _BATCH_ENDPOINT_OK = False
                    else:
                        raise
            # Legacy path.
            out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
            for i, t in enumerate(texts):
                v = _embed_one_legacy(t)
                if v.shape[0] != out.shape[1]:
                    out = np.zeros((len(texts), v.shape[0]), dtype=np.float32)
                out[i] = v
            return out
        except (requests.RequestException, RuntimeError) as e:
            last_exc = e
            if attempt == retries - 1:
                raise
            time.sleep(backoff ** attempt)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Embed a list of *documents* — applies the document task prefix."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    prefixed = [EMBED_DOC_PREFIX + t for t in texts]
    parts: list[np.ndarray] = []
    for i in range(0, len(prefixed), EMBED_BATCH_SIZE):
        chunk = prefixed[i : i + EMBED_BATCH_SIZE]
        parts.append(_embed_chunk(chunk))
    out = np.vstack(parts)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def embed_query(text: str) -> np.ndarray:
    """Embed a *query* — applies the query task prefix."""
    v = _embed_chunk([EMBED_QUERY_PREFIX + text])[0]
    n = np.linalg.norm(v)
    return v if n == 0 else v / n
