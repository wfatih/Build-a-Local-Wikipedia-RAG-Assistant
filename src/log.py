"""Structured (JSON-line) logging.

Every RAG turn writes one line to `data/logs/rag.jsonl` containing:
    timestamp, request_id, model, query, route, retrieve_ms, generate_ms,
    grounding_ms, cached, n_chunks, entities, top_entity_score.

Latency dashboards and offline retrieval-quality scoring read this file.
The format is intentionally append-only NDJSON so it's safe to tail and
parse line-by-line.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from src.config import LOG_DIR


LOG_FILE = LOG_DIR / "rag.jsonl"


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(event: str, **fields: Any) -> None:
    """Append a single JSON line. Failures are swallowed — we never want
    logging to break user requests."""
    rec = {"ts": time.time(), "event": event, **fields}
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def read_recent(n: int = 200) -> list[dict]:
    """Read the last `n` events for the dashboard."""
    if not LOG_FILE.exists():
        return []
    try:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-n:]
    except Exception:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out
