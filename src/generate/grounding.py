"""Self-grounding pass: ask the LLM to verify each sentence of the just-
produced answer is supported by the retrieved context. Drop or rewrite
unsupported sentences. Costs one extra LLM call (≈ doubles latency) so it
ships behind a UI toggle.
"""
from __future__ import annotations

import re

from src.generate.llm import chat


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9\"'(])")
GROUNDING_PROMPT = (
    "You are a strict fact-checker. For EACH sentence I list, decide "
    "whether the provided context items support its factual content.\n"
    "Respond with one line per sentence in the form `N: yes` or `N: no` "
    "where N is the 1-based sentence index. No prose, no explanations."
)


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _parse_verdicts(reply: str, n: int) -> list[bool]:
    out = [True] * n  # default: trust unless the checker explicitly says no
    for line in reply.splitlines():
        m = re.match(r"\s*(\d+)\s*[:.\-]\s*(yes|no|y|n|true|false|t|f)", line, re.IGNORECASE)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        verdict = m.group(2).lower()[0] in ("y", "t")
        if 0 <= idx < n:
            out[idx] = verdict
    return out


def self_check(query: str, answer: str, context_block: str, model: str) -> str:
    """Return either the original answer (if all sentences supported) or a
    pruned version with unsupported sentences removed. If too many
    sentences fail, fall back to the canonical refusal sentence.
    """
    if not answer.strip():
        return answer
    sentences = _split_sentences(answer)
    if len(sentences) <= 1:
        # Nothing to prune sentence-wise; trust the answer.
        return answer

    enumerated = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    user = (
        f"Question: {query}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Sentences to verify:\n{enumerated}\n\n"
        "Reply only with `N: yes` / `N: no` lines."
    )
    try:
        reply = chat(
            [{"role": "system", "content": GROUNDING_PROMPT},
             {"role": "user", "content": user}],
            model=model,
            options={"temperature": 0.0, "num_ctx": 4096},
        )
    except Exception:
        return answer  # fail-open — checker errors must not break answers

    verdicts = _parse_verdicts(reply, len(sentences))
    kept = [s for s, v in zip(sentences, verdicts) if v]
    dropped = len(sentences) - len(kept)
    if not kept:
        return "I don't know based on the provided context."
    # If the model dropped majority, treat the remainder as unreliable.
    if dropped > len(sentences) // 2 and len(kept) <= 1:
        return "I don't know based on the provided context."
    return " ".join(kept)
