"""Direct Wikipedia REST API ingestion. No wrapper libraries — only `requests`."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import requests

from src.config import RAW_DIR, WIKI_API, WIKI_USER_AGENT


@dataclass
class WikiDoc:
    entity: str
    title: str
    type: str  # "person" | "place"
    url: str
    sections: list[dict]  # [{"heading": str, "level": int, "text": str}, ...]
    fetched_at: float


def _slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip().lower()
    return re.sub(r"[-\s]+", "_", s)


def _api_get(params: dict, retries: int = 3, backoff: float = 1.5) -> dict:
    headers = {"User-Agent": WIKI_USER_AGENT}
    for attempt in range(retries):
        try:
            r = requests.get(WIKI_API, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(backoff ** attempt)
    return {}


def _resolve_title(query: str) -> tuple[str, str]:
    """Resolve a free-form name to a canonical Wikipedia title and URL."""
    data = _api_get({
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "srprop": "",
    })
    hits = data.get("query", {}).get("search", [])
    if not hits:
        raise ValueError(f"No Wikipedia article found for: {query}")
    title = hits[0]["title"]
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    return title, url


def _fetch_extract(title: str) -> str:
    """Plain-text extract for the full article (no HTML)."""
    data = _api_get({
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    })
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        if "extract" in page:
            return page["extract"] or ""
    return ""


_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
_CITE_BRACKET_RE = re.compile(r"\[(?:\d+|note\s*\d+|citation needed|n\s*\d+|nb\s*\d+|[a-z])\]", re.IGNORECASE)
_PARENTHETICAL_REF_RE = re.compile(r"\((?:see|cf\.|see also)[^)]{0,80}\)", re.IGNORECASE)
_REPEATED_WS_RE = re.compile(r"[ \t]{2,}")


def _strip_wiki_artifacts(text: str) -> str:
    """Remove Wikipedia editorial markers (citation numbers, [note 3],
    [citation needed], "(see also …)", etc). They're noise for retrieval —
    BM25 will boost any chunk that happens to mention "[1]" otherwise.
    """
    text = _CITE_BRACKET_RE.sub("", text)
    text = _PARENTHETICAL_REF_RE.sub("", text)
    # Normalise whitespace introduced by the deletions.
    text = _REPEATED_WS_RE.sub(" ", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    return text


def _split_sections(extract: str) -> list[dict]:
    """The extracts API returns sections as plain text with no markup; split by
    blank-line transitions and treat lines that look like '== Heading ==' as section
    headers if present. Most extracts use simple line headings — fall back to the
    'header line followed by blank line' heuristic.
    """
    sections: list[dict] = [{"heading": "Introduction", "level": 1, "text": ""}]
    lines = extract.splitlines()
    i = 0
    n = len(lines)
    cur = sections[0]
    buf: list[str] = []
    while i < n:
        line = lines[i]
        # Heuristic: a heading line is short, not blank, followed by blank line,
        # and not ending with sentence punctuation.
        stripped = line.strip()
        is_heading = (
            0 < len(stripped) <= 80
            and not stripped.endswith((".", "!", "?", ":", ",", ";"))
            and (i + 1 >= n or lines[i + 1].strip() == "")
            and (i == 0 or lines[i - 1].strip() == "")
            and not stripped.startswith(("•", "-", "*"))
        )
        if is_heading and buf:
            cur["text"] = "\n".join(buf).strip()
            buf = []
            cur = {"heading": stripped, "level": 2, "text": ""}
            sections.append(cur)
            i += 1
            continue
        if is_heading and not buf and len(sections) == 1:
            cur["heading"] = stripped
            i += 1
            continue
        buf.append(line)
        i += 1
    cur["text"] = "\n".join(buf).strip()
    return [s for s in sections if s["text"]]


def _doc_path(entity: str, type_: str) -> Path:
    return RAW_DIR / type_ / f"{_slugify(entity)}.json"


def fetch_one(entity: str, type_: str, force: bool = False) -> WikiDoc:
    out = _doc_path(entity, type_)
    if out.exists() and not force:
        with out.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return WikiDoc(**d)
    title, url = _resolve_title(entity)
    extract = _fetch_extract(title)
    if not extract.strip():
        raise ValueError(f"Empty extract for {entity} -> {title}")
    extract = _strip_wiki_artifacts(extract)
    sections = _split_sections(extract)
    doc = WikiDoc(
        entity=entity,
        title=title,
        type=type_,
        url=url,
        sections=sections,
        fetched_at=time.time(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(asdict(doc), f, ensure_ascii=False, indent=2)
    return doc


def fetch_all(entities: Iterable[str], type_: str, force: bool = False) -> list[WikiDoc]:
    docs = []
    for e in entities:
        try:
            doc = fetch_one(e, type_, force=force)
            docs.append(doc)
            print(f"  [OK] {type_:6s}  {e}  ->  {doc.title}  ({sum(len(s['text']) for s in doc.sections)} chars)")
        except Exception as ex:  # noqa: BLE001
            print(f"  [FAIL] {type_:6s}  {e}  ->  {ex}")
    return docs


def read_entity_list(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out
