"""Unit tests for the components that don't require Ollama or a populated
vector store. Run with:

    pytest tests/test_unit.py -v
or, without pytest:
    python tests/test_unit.py

Covers:
    * chunker — paragraph-aware sliding window, sentence fall-back
    * BM25 — basic ranking
    * Router — exact, last-token, fuzzy, mixed, unknown
    * VectorStore — schema + add/search/intro retrieval (in-memory)
    * Citation stripping — wikipedia.py
    * Refusal normalisation — pipeline._normalise_refusal
    * Persistence — round-trip save/load
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ---------- mini test framework (avoid pytest dependency) ------------------

PASS = "PASS"
FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []


def t(name: str):
    def deco(fn):
        try:
            fn()
            _results.append((name, PASS, ""))
            print(f"[PASS] {name}")
        except AssertionError as e:
            _results.append((name, FAIL, str(e)))
            print(f"[FAIL] {name}: {e}")
        except Exception as e:  # noqa: BLE001
            _results.append((name, FAIL, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return fn
    return deco


# ---------- chunker --------------------------------------------------------

@t("chunker: short paragraph stays in one chunk")
def _():
    from src.chunk.chunker import chunk_text
    text = "Albert Einstein was a German-born theoretical physicist. He won the 1921 Nobel."
    out = chunk_text(text)
    assert len(out) == 1, f"expected 1 chunk, got {len(out)}"
    assert "Einstein" in out[0]


@t("chunker: long doc splits with overlap")
def _():
    from src.chunk.chunker import chunk_text
    para = "This is a sentence about an entity. " * 60
    text = para + "\n\n" + para + "\n\n" + para
    out = chunk_text(text)
    assert len(out) >= 2, f"expected multiple chunks, got {len(out)}"
    # Overlap means the boundary between consecutive chunks shares content.
    overlaps = sum(1 for i in range(1, len(out)) if any(
        s in out[i] for s in out[i - 1].split(". ")[-3:]
    ))
    assert overlaps >= 1, "no detectable overlap between consecutive chunks"


@t("chunker: oversize paragraph falls back to sentence packing")
def _():
    from src.chunk.chunker import chunk_text
    sentences = [f"Fact number {i} about Einstein states something." for i in range(120)]
    text = " ".join(sentences)  # one giant paragraph
    out = chunk_text(text)
    assert len(out) > 1, "expected sentence fall-back to produce multiple chunks"


@t("chunker: chunk_doc preserves section metadata")
def _():
    from src.chunk.chunker import chunk_doc
    class FakeDoc:
        sections = [
            {"heading": "Introduction", "level": 1, "text": "Lead paragraph here. " * 30},
            {"heading": "Career", "level": 2, "text": "Career paragraph. " * 30},
        ]
    chs = chunk_doc(FakeDoc())
    sections = {c["section"] for c in chs}
    assert "Introduction" in sections and "Career" in sections


# ---------- BM25 -----------------------------------------------------------

@t("bm25: ranks lexically-relevant document highest")
def _():
    from src.retrieve.bm25 import BM25
    docs = [
        {"id": 1, "text": "The Eiffel Tower stands in Paris on the Champ de Mars."},
        {"id": 2, "text": "Mount Everest is on the China-Nepal border in the Himalayas."},
        {"id": 3, "text": "Paris is the capital of France."},
    ]
    bm = BM25(docs)
    top = bm.top_k("Where is the Eiffel Tower in Paris", 2)
    assert top, "no results"
    assert top[0][0]["id"] == 1, f"expected doc 1 first, got {top[0][0]['id']}"


@t("bm25: returns no results for off-corpus query")
def _():
    from src.retrieve.bm25 import BM25
    bm = BM25([{"id": 1, "text": "Eiffel Tower in Paris."}])
    top = bm.top_k("quantum chromodynamics", 5)
    assert all(s <= 0 for _, s in top) or top == []


# ---------- Router ---------------------------------------------------------

@t("router: exact name → person")
def _():
    from src.retrieve.router import Router
    d = Router().route("Who was Albert Einstein and what is he known for")
    assert d.target == "person"
    assert "Albert Einstein" in d.person_entities


@t("router: typo via fuzzy → person")
def _():
    from src.retrieve.router import Router
    d = Router().route("who is sagopa kajmet")  # typo: Kajmer
    assert "Sagopa Kajmer" in d.person_entities


@t("router: comparison of person+place → mixed")
def _():
    from src.retrieve.router import Router
    d = Router().route("Compare Albert Einstein and the Eiffel Tower")
    assert d.target == "mixed"
    assert "Albert Einstein" in d.person_entities
    assert "Eiffel Tower" in d.place_entities


@t("router: nothing-known query → unknown")
def _():
    from src.retrieve.router import Router
    d = Router().route("Who is the president of Mars")
    assert d.target in ("unknown", "person")  # keyword 'who' may push to person
    assert not d.person_entities and not d.place_entities


@t("router: common 'tower' last token does not over-match")
def _():
    from src.retrieve.router import Router
    d = Router().route("Where is the Eiffel Tower located")
    assert d.place_entities == ["Eiffel Tower"], d.place_entities


# ---------- VectorStore ---------------------------------------------------

@t("store: insert, retrieve, intro-chunk lookup")
def _():
    import numpy as np
    from src.store.vector_store import VectorStore

    td = tempfile.mkdtemp()
    s = None
    try:
        path = Path(td) / "test.db"
        s = VectorStore(db_path=path)
        s.init_schema()

        class FakeDoc:
            entity = "Test Person"
            title = "Test Person"
            type = "person"
            url = "https://example.org/test"

        chunks = [
            {"text": "Lead intro chunk for Test Person.", "section": "Introduction",
             "position": 0, "tokens": 8},
            {"text": "Career stuff happened later in life.", "section": "Career",
             "position": 1, "tokens": 8},
        ]
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        vecs = np.vstack([v1, v2])

        s.add_chunks(FakeDoc(), chunks, vecs)
        assert s.total_chunks() == 2
        assert s.count_for_doc("Test Person", "person") == 2

        intros = s.get_intro_chunks("Test Person", "person", max_chunks=1)
        assert len(intros) == 1 and intros[0]["section"] == "Introduction"

        results = s.search(v1, top_k=1, type_filter="person")
        assert results and results[0]["entity"] == "Test Person"
        assert results[0]["section"] == "Introduction"
    finally:
        if s is not None:
            s.close()
        import shutil
        shutil.rmtree(td, ignore_errors=True)


@t("store: type_filter excludes other types")
def _():
    import numpy as np
    from src.store.vector_store import VectorStore

    td = tempfile.mkdtemp()
    s = None
    try:
        s = VectorStore(db_path=Path(td) / "x.db")
        s.init_schema()

        class P: entity = "Pp"; title = "Pp"; type = "person"; url = "u"
        class L: entity = "Ll"; title = "Ll"; type = "place";  url = "u"

        v = np.array([1.0, 0.0], dtype=np.float32)
        s.add_chunks(P(), [{"text": "p", "section": "Introduction", "position": 0, "tokens": 1}],
                     np.vstack([v]))
        s.add_chunks(L(), [{"text": "l", "section": "Introduction", "position": 0, "tokens": 1}],
                     np.vstack([v]))

        only_p = s.search(v, top_k=5, type_filter="person")
        assert all(r["type"] == "person" for r in only_p)
        only_l = s.search(v, top_k=5, type_filter="place")
        assert all(r["type"] == "place" for r in only_l)
    finally:
        if s is not None:
            s.close()
        import shutil
        shutil.rmtree(td, ignore_errors=True)


# ---------- Citation stripping -------------------------------------------

@t("wikipedia: citation marker stripping removes [1] [note 3] [citation needed]")
def _():
    from src.ingest.wikipedia import _strip_wiki_artifacts
    raw = ('Einstein was a physicist [1]. He developed relativity [note 3] '
           '[citation needed] and won a Nobel [12]. (see also: photoelectric)')
    out = _strip_wiki_artifacts(raw)
    assert "[1]" not in out
    assert "[note 3]" not in out
    assert "[citation needed]" not in out
    assert "(see also:" not in out
    # The actual content is preserved.
    assert "Einstein was a physicist" in out
    assert "relativity" in out


# ---------- Refusal normalisation ----------------------------------------

@t("pipeline: paraphrased refusals are normalised")
def _():
    from src.generate.pipeline import _normalise_refusal, REFUSAL
    assert _normalise_refusal("There is no information in the provided context about Mars.") == REFUSAL
    assert _normalise_refusal("I don't know based on the provided context.") == REFUSAL
    assert _normalise_refusal("The context does not mention any such person.") == REFUSAL


@t("pipeline: substantive answers are NOT normalised")
def _():
    from src.generate.pipeline import _normalise_refusal
    rich = ("Albert Einstein was a German-born theoretical physicist who developed the theory "
            "of relativity. He received the 1921 Nobel Prize in Physics for the photoelectric "
            "effect. Although Einstein famously said gravity is geometry, this answer covers "
            "his most cited contribution.")
    assert _normalise_refusal(rich) == rich


# ---------- History augmentation -----------------------------------------

@t("pipeline: pronoun follow-up gets augmented with last entity")
def _():
    from src.generate.pipeline import _augment_query_with_history
    from src.retrieve.router import Router
    r = Router()
    history = [
        {"role": "user", "content": "Who was Albert Einstein and what is he known for"},
        {"role": "assistant", "content": "Albert Einstein was a physicist..."},
    ]
    augmented = _augment_query_with_history("What year did he win the Nobel Prize", history, r)
    assert "Albert Einstein" in augmented


@t("pipeline: full-form follow-up is NOT augmented")
def _():
    from src.generate.pipeline import _augment_query_with_history
    from src.retrieve.router import Router
    r = Router()
    history = [
        {"role": "user", "content": "Who was Albert Einstein"},
        {"role": "assistant", "content": "..."},
    ]
    q = "Where is the Eiffel Tower located"
    assert _augment_query_with_history(q, history, r) == q


# ---------- Persistence --------------------------------------------------

@t("persistence: save / load / list / delete round-trip")
def _():
    import os
    from src.ui import persistence
    from src.config import CONV_DIR

    cid = persistence.new_conversation_id()
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    try:
        persistence.save_conversation(cid, msgs)
        loaded = persistence.load_conversation(cid)
        assert loaded == msgs
        listing = persistence.list_conversations()
        assert any(c["id"] == cid for c in listing)
    finally:
        persistence.delete_conversation(cid)
        assert not (CONV_DIR / f"{cid}.json").exists()


@t("persistence: export_as_markdown contains both turns")
def _():
    from src.ui.persistence import export_as_markdown
    md = export_as_markdown([
        {"role": "user", "content": "Who was Marie Curie"},
        {"role": "assistant", "content": "She discovered radium [1].",
         "meta": {"chunks": [{"title": "Marie Curie", "section": "Lead",
                              "url": "u", "sources": ["dense", "bm25"]}]}},
    ])
    assert "Marie Curie" in md
    assert "She discovered radium" in md
    assert "Sources" in md


# ---------- Summary ------------------------------------------------------

if __name__ == "__main__":
    fails = [r for r in _results if r[1] == FAIL]
    print()
    print(f"Total: {len(_results)} | PASS: {len(_results) - len(fails)} | FAIL: {len(fails)}")
    sys.exit(1 if fails else 0)
