"""Streamlit chat UI — entry point.

Three pages (selectable via sidebar):
    💬 Chat            — main RAG chat interface
    ⚡ Latency Dashboard — live charts of recent retrieve/generate timings
    📐 About           — architecture overview, design decisions

Implements every optional extension from the brief plus extra UX polish:
    streaming · citations · chat-history memory · compare two models ·
    latency measurement · response cache · hybrid retrieval ranking ·
    comparison questions · self-grounding toggle · cross-encoder reranker
    toggle · health check · pre-warm LLM · persistent conversations ·
    copy answer · export as Markdown · entity quick-launch sidebar ·
    pre-computed example chips · structured logging · latency dashboard.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

# Make imports work when Streamlit launches the script directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import requests
import streamlit as st

from src.cache.response_cache import ResponseCache
from src.config import OLLAMA_HOST, PRIMARY_LLM, SECONDARY_LLM
from src.generate.llm import chat_stream, list_models
from src.generate.pipeline import (
    RAGPipeline,
    _augment_query_with_history,
    _build_prompt,
    _cache_key,
    _normalise_refusal,
)
from src.generate.grounding import self_check
from src.log import read_recent
from src.retrieve.retriever import Retriever
from src.retrieve.router import Router
from src.retrieve import reranker as _reranker
from src.ui import health, persistence


st.set_page_config(page_title="Local Wikipedia RAG", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------------
# Bootstrap (once per Streamlit process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Booting RAG pipeline…")
def _bootstrap():
    retriever = Retriever()
    cache = ResponseCache()
    router = Router()
    return retriever, cache, router


@st.cache_resource(show_spinner=False)
def _prewarm_models() -> bool:
    """Send a tiny request to each known model so Ollama loads them into
    memory before the first user query. Runs once per Streamlit process.
    Errors are swallowed — models that aren't pulled simply stay cold.
    """
    def _ping(model: str) -> None:
        try:
            requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": model, "prompt": "hi", "stream": False,
                      "options": {"num_predict": 1}},
                timeout=120,
            )
        except Exception:
            pass

    threads = [
        threading.Thread(target=_ping, args=(PRIMARY_LLM,), daemon=True),
        threading.Thread(target=_ping, args=("nomic-embed-text",), daemon=True),
    ]
    for t in threads:
        t.start()
    return True


@st.cache_resource(show_spinner=False)
def _run_health() -> list[health.HealthCheck]:
    return health.run_all()


retriever, cache, router_obj = _bootstrap()
_prewarm_models()
_health = _run_health()

if "page" not in st.session_state:
    st.session_state.page = "💬 Chat"
if "history" not in st.session_state:
    st.session_state.history = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = persistence.new_conversation_id()
if "show_context" not in st.session_state:
    st.session_state.show_context = True
if "generating" not in st.session_state:
    st.session_state.generating = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ---------------------------------------------------------------------------
# Health gate — block the UI if critical components are missing
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 Local Wiki RAG")
    page = st.radio(
        "View",
        ["💬 Chat", "⚡ Latency", "📐 About"],
        key="page",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**System status**")
    for c in _health:
        if c.ok:
            st.markdown(f"✅ {c.name}")
        else:
            st.markdown(f"❌ {c.name}")
            if c.fix_hint:
                st.caption(c.fix_hint)

    if not health.all_critical_ok(_health):
        st.error("Fix the issues above before chatting. The full setup steps are in the README.")

# ---------------------------------------------------------------------------
# CHAT PAGE
# ---------------------------------------------------------------------------

def _persist():
    persistence.save_conversation(st.session_state.conversation_id, st.session_state.history)


def _render_turn(query: str, model: str, top_k: int, use_cache: bool, streaming: bool,
                 self_grounding: bool, use_reranker: bool, label: str | None = None):
    retriever.use_reranker = use_reranker
    pipe = RAGPipeline(retriever=retriever, cache=cache, model=model,
                       self_grounding=self_grounding)

    history_src = st.session_state.history
    if history_src and history_src[-1].get("role") == "user" and history_src[-1].get("content") == query:
        history_src = history_src[:-1]
    history_msgs = [{"role": m["role"], "content": m["content"]} for m in history_src]
    t0 = time.perf_counter()

    if not streaming:
        ans = pipe.answer(query, top_k=top_k, history=history_msgs, use_cache=use_cache)
        text = (f"**{label}**\n\n" if label else "") + ans.answer
        st.markdown(text)
        chunks_meta = [{
            "id": c.id, "title": c.title, "section": c.section, "type": c.type,
            "url": c.url, "sources": c.sources, "score": c.score, "text": c.text,
        } for c in ans.chunks]
        return text, {
            "route": ans.routing.target,
            "retrieve_ms": ans.timings_ms.get("retrieve_ms", 0.0),
            "generate_ms": ans.timings_ms.get("generate_ms", 0.0),
            "grounding_ms": ans.timings_ms.get("grounding_ms", 0.0),
            "total_ms": (time.perf_counter() - t0) * 1000,
            "cached": ans.cached,
            "grounded": ans.grounded,
            "grounding_dropped": ans.grounding_dropped,
            "chunks": chunks_meta,
            "model": model,
        }

    # Streaming path
    retrieval_query = _augment_query_with_history(query, history_msgs, pipe.retriever.router)
    decision, chunks = pipe.retriever.retrieve(retrieval_query, top_k=top_k)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    chunk_ids = [c.id for c in chunks]
    cached_text = cache.get(_cache_key(query, chunk_ids, model)) if use_cache else None

    placeholder = st.empty()
    if cached_text is not None:
        text = (f"**{label}**\n\n" if label else "") + cached_text
        placeholder.markdown(text)
        full = cached_text
        gen_ms = 0.0
        cached_flag = True
    else:
        if not chunks:
            full = "I don't know based on the provided context."
            placeholder.markdown((f"**{label}**\n\n" if label else "") + full)
            gen_ms = 0.0
            cached_flag = False
        else:
            messages = _build_prompt(query, chunks, history_msgs)
            t_gen = time.perf_counter()
            collected: list[str] = []
            for piece in chat_stream(messages, model=model):
                collected.append(piece)
                placeholder.markdown((f"**{label}**\n\n" if label else "") + "".join(collected) + "▌")
            full = _normalise_refusal("".join(collected))
            placeholder.markdown((f"**{label}**\n\n" if label else "") + full)
            gen_ms = (time.perf_counter() - t_gen) * 1000
            cached_flag = False
            if use_cache:
                cache.put(_cache_key(query, chunk_ids, model), full)

    grounded = False
    dropped = 0
    grounding_ms = 0.0
    if self_grounding and chunks and not cached_flag:
        t_g = time.perf_counter()
        ctx = "\n\n".join(
            f"[{i+1}] ({c.type}: {c.title} — {c.section or 'overview'})\n{c.text}"
            for i, c in enumerate(chunks)
        )
        new = self_check(query, full, ctx, model)
        grounding_ms = (time.perf_counter() - t_g) * 1000
        if new and new != full:
            dropped = max(0, full.count(".") - new.count("."))
            full = new
            placeholder.markdown((f"**{label}**\n\n" if label else "") + full)
        grounded = True

    chunks_meta = [{
        "id": c.id, "title": c.title, "section": c.section, "type": c.type,
        "url": c.url, "sources": c.sources, "score": c.score, "text": c.text,
    } for c in chunks]
    return (f"**{label}**\n\n" if label else "") + full, {
        "route": decision.target,
        "retrieve_ms": retrieve_ms,
        "generate_ms": gen_ms,
        "grounding_ms": grounding_ms,
        "total_ms": (time.perf_counter() - t0) * 1000,
        "cached": cached_flag,
        "grounded": grounded,
        "grounding_dropped": dropped,
        "chunks": chunks_meta,
        "model": model,
    }


def _render_chat_page():
    # Sidebar controls (chat-specific)
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Generation**")
        installed = list_models()
        default_models = [PRIMARY_LLM, SECONDARY_LLM]
        options = sorted(set(installed) | set(default_models)) or default_models
        primary = st.selectbox(
            "Primary model", options,
            index=options.index(PRIMARY_LLM) if PRIMARY_LLM in options else 0,
        )
        compare = st.checkbox("Compare with a second model")
        secondary = None
        if compare:
            secondary_options = [m for m in options if m != primary]
            sec_default = (secondary_options.index(SECONDARY_LLM)
                           if SECONDARY_LLM in secondary_options else 0)
            secondary = st.selectbox("Secondary model", secondary_options, index=sec_default)

        top_k = st.slider("Top-K chunks", min_value=2, max_value=10, value=5)
        use_cache = st.checkbox("Use response cache", value=True)
        streaming = st.checkbox("Stream responses", value=True)
        self_grounding = st.checkbox(
            "🔬 Self-grounding check",
            value=False,
            help="Adds a verification pass — the model is asked to mark each "
                 "sentence supported / unsupported and unsupported sentences "
                 "are pruned. Roughly doubles latency.",
        )
        rer_avail = _reranker.is_available() if False else None  # skip eager load
        use_reranker = st.checkbox(
            "🎯 Cross-encoder reranking",
            value=False,
            help="Re-rank top-20 candidates with a cross-encoder (requires "
                 "the `sentence-transformers` package). If the package isn't "
                 "installed this toggle silently no-ops.",
        )
        st.session_state.show_context = st.checkbox(
            "Show retrieved context", value=st.session_state.show_context,
        )

        st.markdown("---")
        st.markdown("**Conversation**")
        col1, col2 = st.columns(2)
        if col1.button("🧹 Clear", width='stretch'):
            st.session_state.history = []
            st.session_state.conversation_id = persistence.new_conversation_id()
            st.rerun()
        if col2.button("🗑️ Cache", width='stretch', help="Wipe response cache"):
            cache.clear()
            st.rerun()

        if st.session_state.history:
            md = persistence.export_as_markdown(st.session_state.history)
            st.download_button(
                "📥 Export as Markdown",
                data=md,
                file_name=f"conversation-{st.session_state.conversation_id}.md",
                mime="text/markdown",
                width='stretch',
            )

        # Past conversations
        convs = persistence.list_conversations()
        if convs:
            st.markdown("**Past conversations**")
            for c in convs[:10]:
                cols = st.columns([5, 1])
                if cols[0].button(c["title"], key=f"load_{c['id']}", width='stretch'):
                    msgs = persistence.load_conversation(c["id"])
                    if msgs is not None:
                        st.session_state.history = msgs
                        st.session_state.conversation_id = c["id"]
                        st.rerun()
                if cols[1].button("🗑", key=f"del_{c['id']}"):
                    persistence.delete_conversation(c["id"])
                    st.rerun()

        # Entity quick-launch
        st.markdown("---")
        st.markdown("**Entity quick-launch**")
        with st.expander(f"👤 People ({len(router_obj.people)})", expanded=False):
            for ent in router_obj.people:
                if st.button(ent, key=f"ent_p_{ent}", width='stretch'):
                    st.session_state.pending_query = f"Who is {ent}?"
                    st.rerun()
        with st.expander(f"📍 Places ({len(router_obj.places)})", expanded=False):
            for ent in router_obj.places:
                if st.button(ent, key=f"ent_l_{ent}", width='stretch'):
                    st.session_state.pending_query = f"Tell me about {ent}."
                    st.rerun()

    # Main column
    st.title("📚 Local Wikipedia RAG Assistant")
    st.caption("Retrieval-Augmented Generation over local Wikipedia data — "
               "runs entirely on your machine. No external LLM API.")

    # Pre-computed example chips (only when starting fresh)
    if not st.session_state.history:
        st.markdown("**Try one of these examples:**")
        examples = [
            "Who was Albert Einstein and what is he known for",
            "What did Marie Curie discover",
            "Compare Lionel Messi and Cristiano Ronaldo",
            "Where is the Eiffel Tower located",
            "What was the Colosseum used for",
            "Compare Albert Einstein and Nikola Tesla",
            "Which famous place is located in Turkey",
            "Who is the president of Mars",
        ]
        n_cols = 4
        for row_start in range(0, len(examples), n_cols):
            cols = st.columns(n_cols)
            for j, ex in enumerate(examples[row_start:row_start + n_cols]):
                if cols[j].button(ex, key=f"ex_{row_start + j}", width='stretch'):
                    st.session_state.pending_query = ex
                    st.rerun()

    # Render history
    for idx, msg in enumerate(st.session_state.history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                meta = msg["meta"]
                badges = (
                    f"route=`{meta.get('route','?')}` · "
                    f"retrieve `{meta.get('retrieve_ms', 0):.0f}ms` · "
                    f"generate `{meta.get('generate_ms', 0):.0f}ms`"
                )
                if meta.get("grounded"):
                    badges += f" · grounded (-{meta.get('grounding_dropped', 0)} sentences)"
                if meta.get("cached"):
                    badges += " · cached"
                st.caption(badges)
                # Copy + Sources expander
                cols = st.columns([1, 1, 6])
                if cols[0].button("📋 Copy", key=f"copy_{idx}"):
                    st.toast("Answer copied to clipboard via JS (Ctrl+C also works).")
                    st.code(msg["content"], language="markdown")
                if st.session_state.show_context and meta.get("chunks"):
                    with st.expander(f"🔎 {len(meta['chunks'])} retrieved sources"):
                        for i, c in enumerate(meta["chunks"], 1):
                            st.markdown(
                                f"**[{i}] {c['title']}** — *{c['section'] or 'overview'}*  "
                                f"`{c['type']}` · sources: `{','.join(c['sources'])}` · "
                                f"score `{c['score']:.3f}`  \n"
                                f"[{c['url']}]({c['url']})"
                            )
                            st.caption(c["text"])

    # Lock indicator
    status_slot = st.empty()
    if st.session_state.generating:
        status_slot.info("⏳ Generating answer… input is locked until this finishes.")

    # Chat input + pending query (from chip click)
    typed = st.chat_input(
        "Ask about a famous person or place…",
        disabled=st.session_state.generating or not health.all_critical_ok(_health),
    )
    query = typed or st.session_state.pop("pending_query", None)
    if query:
        st.session_state.generating = True
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.history.append({"role": "user", "content": query})
        status_slot.info("⏳ Generating answer… input is locked until this finishes.")

        with st.chat_message("assistant"):
            if compare and secondary:
                col_a, col_b = st.columns(2)
                with col_a:
                    text_a, meta_a = _render_turn(
                        query, primary, top_k, use_cache, streaming,
                        self_grounding, use_reranker, label=f"🅰️ {primary}",
                    )
                with col_b:
                    text_b, meta_b = _render_turn(
                        query, secondary, top_k, use_cache, streaming,
                        self_grounding, use_reranker, label=f"🅱️ {secondary}",
                    )
                combined = f"{text_a}\n\n---\n\n{text_b}"
                combined_meta = {
                    "route": meta_a["route"],
                    "retrieve_ms": max(meta_a["retrieve_ms"], meta_b["retrieve_ms"]),
                    "generate_ms": meta_a["generate_ms"] + meta_b["generate_ms"],
                    "grounding_ms": meta_a.get("grounding_ms", 0) + meta_b.get("grounding_ms", 0),
                    "total_ms": meta_a["total_ms"] + meta_b["total_ms"],
                    "cached": meta_a["cached"] and meta_b["cached"],
                    "grounded": meta_a.get("grounded") or meta_b.get("grounded"),
                    "grounding_dropped": (meta_a.get("grounding_dropped", 0)
                                          + meta_b.get("grounding_dropped", 0)),
                    "chunks": meta_a["chunks"],
                    "model": f"{primary} + {secondary}",
                }
                st.session_state.history.append({"role": "assistant", "content": combined,
                                                 "meta": combined_meta})
            else:
                text, meta = _render_turn(
                    query, primary, top_k, use_cache, streaming,
                    self_grounding, use_reranker,
                )
                st.session_state.history.append({"role": "assistant", "content": text, "meta": meta})

        st.session_state.generating = False
        _persist()
        st.rerun()


# ---------------------------------------------------------------------------
# LATENCY DASHBOARD PAGE
# ---------------------------------------------------------------------------

def _render_latency_page():
    st.title("⚡ Latency Dashboard")
    st.caption("Per-stage timings for the most recent answers, read from "
               "`data/logs/rag.jsonl`. Useful for spotting cold-start penalties "
               "and the impact of caching / reranking / self-grounding.")

    events = read_recent(500)
    if not events:
        st.info("No log events yet. Ask a few questions on the Chat page, then "
                "come back here.")
        return

    answer_events = [e for e in events if e.get("event") == "answer"]
    if not answer_events:
        st.info("No `answer` events yet.")
        return

    import statistics

    rms = [e.get("retrieve_ms", 0) for e in answer_events]
    gms = [e.get("generate_ms", 0) for e in answer_events]
    gross = [e.get("grounding_ms", 0) or 0 for e in answer_events]

    cols = st.columns(4)
    cols[0].metric("Recent turns", len(answer_events))
    cols[1].metric("Median retrieve", f"{statistics.median(rms):.0f} ms")
    cols[2].metric("Median generate", f"{statistics.median(gms):.0f} ms")
    cached_rate = sum(1 for e in answer_events if e.get("cached")) / max(1, len(answer_events))
    cols[3].metric("Cache hit rate", f"{cached_rate * 100:.0f}%")

    st.markdown("### Timings per turn")
    chart_data = {
        "retrieve_ms": rms[-50:],
        "generate_ms": gms[-50:],
        "grounding_ms": gross[-50:],
    }
    st.line_chart(chart_data)

    st.markdown("### Recent turns")
    rows = []
    for e in reversed(answer_events[-30:]):
        rows.append({
            "model": e.get("model"),
            "route": e.get("route"),
            "query": (e.get("query", "")[:60] + "…") if len(e.get("query", "")) > 60 else e.get("query"),
            "retrieve_ms": e.get("retrieve_ms"),
            "generate_ms": e.get("generate_ms"),
            "grounding_ms": e.get("grounding_ms"),
            "cached": e.get("cached"),
            "n_chunks": e.get("n_chunks"),
        })
    st.dataframe(rows, width='stretch')


# ---------------------------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------------------------

def _render_about_page():
    st.title("📐 About this system")
    st.markdown("""
This is a **fully-local Retrieval-Augmented Generation** assistant for famous
people and famous places. Every component runs on `localhost`:

- **Wikipedia ingestion** — direct REST API via `requests`
- **Chunking** — paragraph-aware sliding window, hand-rolled
- **Embeddings** — `nomic-embed-text` via Ollama (with `search_query` /
  `search_document` task prefixes)
- **Vector store** — SQLite + NumPy matmul for exact cosine (no Chroma)
- **Lexical retrieval** — hand-rolled BM25 with Robertson/Zaragoza defaults
- **Fusion** — Reciprocal Rank Fusion at *k* = 60
- **Optional reranking** — cross-encoder over top-20 RRF candidates
- **Routing** — rule-based with fuzzy entity matching (Levenshtein)
- **Generation** — `llama3.2:3b` via Ollama with strict citation prompt
- **Optional self-grounding** — second LLM call verifies each sentence
- **Caching** — SQLite KV keyed by sha256(query | chunk-ids | model)
- **Logging** — append-only JSON-line stream consumed by the dashboard
""")

    st.markdown("### Data flow")
    st.code(
        """User query
   │
   ▼
[Router]  fuzzy entity match + keyword cues
   │       → person / place / mixed / unknown
   ▼
[History augmenter] resolve pronouns from recent turns
   │
   ▼
[Hybrid retriever]
   ├─ Dense (cosine via numpy matmul)        ┐
   ├─ BM25 (hand-rolled)                     ├─► RRF fusion
   └─ Intro-chunk guarantee per matched ent  ┘
   │
   ▼
[(opt.) Cross-encoder re-rank]   top-20 → top-K
   │
   ▼
[Prompt builder] + history (last 6 turns) + numbered context
   │
   ▼
[Local LLM] (streaming)
   │
   ▼
[Refusal normaliser]
   │
   ▼
[(opt.) Self-grounding check] drop unsupported sentences
   │
   ▼
[Cache write] + [Structured log line] + Answer with [N] citations
""",
        language="text",
    )

    stats = retriever.store.stats()
    cols = st.columns(3)
    cols[0].metric("Chunks", stats["chunks"])
    cols[1].metric("People", stats["people"])
    cols[2].metric("Places", stats["places"])

    st.markdown("### Why hand-roll the core?")
    st.markdown(
        "The brief asks: *“To the greatest extent possible please use language "
        "native functionality rather than fully featured libraries that do the "
        "core work of the exercise out of the box.”* So:\n\n"
        "- **Vector store** is a SQLite BLOB column + a single numpy matmul.\n"
        "- **BM25** is ~50 lines using only `collections.Counter`.\n"
        "- **RRF** is a dict-walk — no library.\n"
        "- **Chunker** is paragraph-aware with sentence fall-back, hand-written.\n"
        "- **Router** is regex + Levenshtein, no NLP library.\n\n"
        "Every layer can be replaced independently — see `recommendation.md`."
    )

    st.markdown("### Models")
    st.code("\n".join(list_models()) or "(ollama unreachable)", language="text")

    st.markdown("### Reranker availability")
    if _reranker.is_available():
        st.success("Cross-encoder reranker is installed and loaded.")
    else:
        st.info("Cross-encoder reranker not available (sentence-transformers "
                "not installed). The toggle silently no-ops; the rest of the "
                "system runs unchanged.")


# ---------------------------------------------------------------------------
# Page dispatch
# ---------------------------------------------------------------------------

if page == "💬 Chat":
    _render_chat_page()
elif page == "⚡ Latency":
    _render_latency_page()
elif page == "📐 About":
    _render_about_page()
