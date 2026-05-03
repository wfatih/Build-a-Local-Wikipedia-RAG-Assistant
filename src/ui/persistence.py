"""Persistent chat history. Each conversation is one JSON file under
`data/conversations/<id>.json`. Cheap and human-inspectable."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from src.config import CONV_DIR


def _conv_path(conv_id: str) -> Path:
    return CONV_DIR / f"{conv_id}.json"


def new_conversation_id() -> str:
    return uuid.uuid4().hex[:10]


def list_conversations() -> list[dict]:
    """Most-recent-first list of conversation summaries."""
    out: list[dict] = []
    for p in CONV_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        msgs = d.get("messages", [])
        first_user = next((m["content"] for m in msgs if m.get("role") == "user"), "(empty)")
        out.append({
            "id": d.get("id", p.stem),
            "title": (first_user[:60] + "…") if len(first_user) > 60 else first_user,
            "updated": d.get("updated", p.stat().st_mtime),
            "n_msgs": len(msgs),
        })
    out.sort(key=lambda r: -r["updated"])
    return out


def load_conversation(conv_id: str) -> list[dict] | None:
    p = _conv_path(conv_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("messages", [])
    except Exception:
        return None


def save_conversation(conv_id: str, messages: list[dict]) -> None:
    p = _conv_path(conv_id)
    payload = {
        "id": conv_id,
        "updated": time.time(),
        "messages": messages,
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_conversation(conv_id: str) -> None:
    p = _conv_path(conv_id)
    if p.exists():
        p.unlink()


def export_as_markdown(messages: list[dict]) -> str:
    """Serialize a conversation as a Markdown document the user can save."""
    lines: list[str] = ["# Local Wikipedia RAG — Conversation Export", ""]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if role == "user":
            lines.append(f"### 👤 You")
            lines.append("")
            lines.append(content)
        else:
            lines.append(f"### 🤖 Assistant")
            lines.append("")
            lines.append(content)
            meta = msg.get("meta") or {}
            chunks = meta.get("chunks") or []
            if chunks:
                lines.append("")
                lines.append("<details><summary>Sources</summary>")
                lines.append("")
                for i, c in enumerate(chunks, 1):
                    lines.append(
                        f"{i}. **{c.get('title', '?')}** — *{c.get('section') or 'overview'}* "
                        f"([link]({c.get('url', '')})) — sources: `{','.join(c.get('sources') or [])}`"
                    )
                lines.append("")
                lines.append("</details>")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
