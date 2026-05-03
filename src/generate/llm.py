"""Thin Ollama chat client. Streaming + non-streaming. Hand-rolled HTTP, no
high-level wrappers — keeps the demo honest about what's happening.
"""
from __future__ import annotations

import json
from typing import Iterator

import requests

from src.config import OLLAMA_HOST, PRIMARY_LLM


def chat(
    messages: list[dict],
    model: str = PRIMARY_LLM,
    options: dict | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options or {"temperature": 0.2, "num_ctx": 4096},
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")


def chat_stream(
    messages: list[dict],
    model: str = PRIMARY_LLM,
    options: dict | None = None,
) -> Iterator[str]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options or {"temperature": 0.2, "num_ctx": 4096},
    }
    with requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("done"):
                break
            piece = obj.get("message", {}).get("content", "")
            if piece:
                yield piece


def list_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except requests.RequestException:
        return []
