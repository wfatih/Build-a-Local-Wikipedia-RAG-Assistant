"""Paragraph-aware sliding-window chunker.

Strategy:
    1. Walk sections, then paragraphs (split on blank line).
    2. Greedily pack paragraphs into a chunk while staying near the target token budget.
    3. If a single paragraph is larger than the budget, sentence-split and pack sentences.
    4. Carry an `overlap` tail (in tokens) from each emitted chunk into the next so
       cross-paragraph context survives.

Why not fixed-size? Wikipedia paragraphs are coherent semantic units; respecting
them produces chunks that read as standalone passages and retrieve far better.
"""
from __future__ import annotations

import re
from typing import Iterable

from src.config import APPROX_CHARS_PER_TOKEN, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9\"'(])")


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // APPROX_CHARS_PER_TOKEN)


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _pack(units: Iterable[str], target: int, overlap: int) -> list[str]:
    """Pack a stream of textual units (sentences or paragraphs) into chunks of
    ~target tokens, carrying an overlap of `overlap` tokens between chunks.
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    units = list(units)
    for u in units:
        u_tok = _approx_tokens(u)
        if buf and buf_tokens + u_tok > target:
            chunks.append(" ".join(buf).strip())
            # Build overlap tail by walking back from the end of buf.
            tail: list[str] = []
            tail_tok = 0
            for prev in reversed(buf):
                p_tok = _approx_tokens(prev)
                if tail_tok + p_tok > overlap:
                    break
                tail.insert(0, prev)
                tail_tok += p_tok
            buf = tail[:]
            buf_tokens = sum(_approx_tokens(t) for t in buf)
        buf.append(u)
        buf_tokens += u_tok
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]


def chunk_text(text: str) -> list[str]:
    paragraphs = _split_paragraphs(text)
    expanded: list[str] = []
    for p in paragraphs:
        if _approx_tokens(p) > CHUNK_TARGET_TOKENS:
            expanded.extend(_split_sentences(p))
        else:
            expanded.append(p)
    return _pack(expanded, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS)


def chunk_doc(doc) -> list[dict]:
    """Return a list of chunk dicts with metadata. `doc` is a WikiDoc."""
    out: list[dict] = []
    position = 0
    for sec in doc.sections:
        for piece in chunk_text(sec["text"]):
            out.append({
                "text": piece,
                "section": sec["heading"],
                "position": position,
                "tokens": _approx_tokens(piece),
            })
            position += 1
    return out
