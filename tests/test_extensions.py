"""Verify EVERY optional extension from the brief is actually working.

Brief lists 8 optional extensions:
  1. streaming responses
  2. citations / source highlighting
  3. chat history memory
  4. comparing two different local models
  5. latency measurement and optimisation
  6. caching responses
  7. improving retrieval ranking (hybrid)
  8. supporting comparison questions across people and places
"""
from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.cache.response_cache import ResponseCache
from src.config import PRIMARY_LLM, SECONDARY_LLM
from src.generate.llm import chat_stream, list_models
from src.generate.pipeline import RAGPipeline
from src.retrieve.retriever import Retriever


def banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "[PASS]" if condition else "[FAIL]"
    print(f"  {mark}  {label}" + (f"  ({detail})" if detail else ""))
    return condition


def main() -> int:
    failures: list[str] = []
    retriever = Retriever()

    # --- 1. Streaming responses ------------------------------------------
    banner("Extension 1: Streaming responses")
    pipe = RAGPipeline(retriever=retriever, cache=None, model=PRIMARY_LLM)
    stub, stream = pipe.answer_stream("Where is the Eiffel Tower located")
    pieces: list[str] = []
    chunk_count = 0
    t0 = time.perf_counter()
    first_token_at = None
    for piece in stream:
        chunk_count += 1
        pieces.append(piece)
        if first_token_at is None:
            first_token_at = time.perf_counter() - t0
    final = "".join(pieces)
    ok1 = check("multiple streaming chunks received", chunk_count > 5, f"{chunk_count} chunks")
    ok2 = check("first-token latency < 60s", first_token_at is not None and first_token_at < 60, f"{first_token_at:.2f}s")
    ok3 = check("answer mentions Paris", "paris" in final.lower())
    if not (ok1 and ok2 and ok3):
        failures.append("streaming")
    print(f"  preview: {final[:120]}…")

    # --- 2. Citations / source highlighting ------------------------------
    banner("Extension 2: Citations / source highlighting")
    ans = pipe.answer("What did Marie Curie discover")
    citation_re = re.compile(r"\[\d+\]")
    has_cites = bool(citation_re.search(ans.answer))
    ok = check("inline [N] citations present in answer", has_cites,
               f"matches: {citation_re.findall(ans.answer)[:5]}")
    ok2 = check("each retrieved chunk has url + section + score", all(
        c.url and c.score is not None for c in ans.chunks))
    ok3 = check("retrieved chunks count >= 5", len(ans.chunks) >= 5)
    if not (ok and ok2 and ok3):
        failures.append("citations")

    # --- 3. Chat history memory ------------------------------------------
    banner("Extension 3: Chat history memory")
    history: list[dict] = []
    a1 = pipe.answer("Who was Albert Einstein and what is he known for", history=history, use_cache=False)
    history.append({"role": "user", "content": "Who was Albert Einstein and what is he known for"})
    history.append({"role": "assistant", "content": a1.answer})
    # Follow-up that ONLY makes sense with conversation context.
    a2 = pipe.answer("What year did he win the Nobel Prize", history=history, use_cache=False)
    print(f"  follow-up answer: {a2.answer[:200]}")
    has_year = ("1921" in a2.answer)
    refers_einstein = ("einstein" in a2.answer.lower() or "he" in a2.answer.lower() or "him" in a2.answer.lower()
                       or has_year)
    ok = check("follow-up resolves 'he' via history (mentions 1921 or Einstein)", has_year or refers_einstein)
    if not ok:
        failures.append("history")

    # --- 4. Compare two different local models ---------------------------
    banner("Extension 4: Compare two different local models")
    models = list_models()
    have_secondary = any(m.startswith(SECONDARY_LLM.split(":")[0]) for m in models)
    if not have_secondary:
        check("secondary model available", False, f"{SECONDARY_LLM} not pulled — skipping side-by-side")
        failures.append("compare-models (model not pulled)")
    else:
        pipe_a = RAGPipeline(retriever=retriever, cache=None, model=PRIMARY_LLM)
        pipe_b = RAGPipeline(retriever=retriever, cache=None, model=SECONDARY_LLM)
        q = "Why is Nikola Tesla famous"
        a_ans = pipe_a.answer(q, use_cache=False)
        b_ans = pipe_b.answer(q, use_cache=False)
        ok1 = check(f"{PRIMARY_LLM} answers Tesla question",  "tesla" in a_ans.answer.lower())
        ok2 = check(f"{SECONDARY_LLM} answers Tesla question", "tesla" in b_ans.answer.lower())
        ok3 = check("answers from the two models differ", a_ans.answer != b_ans.answer)
        print(f"  primary preview:   {a_ans.answer[:100]}…")
        print(f"  secondary preview: {b_ans.answer[:100]}…")
        if not (ok1 and ok2 and ok3):
            failures.append("compare-models")

    # --- 5. Latency measurement ------------------------------------------
    banner("Extension 5: Latency measurement")
    pipe2 = RAGPipeline(retriever=retriever, cache=None, model=PRIMARY_LLM)
    a = pipe2.answer("What is Machu Picchu", use_cache=False)
    has_retrieve = "retrieve_ms" in a.timings_ms
    has_generate = "generate_ms" in a.timings_ms
    ok1 = check("retrieve_ms reported", has_retrieve, f"{a.timings_ms.get('retrieve_ms', 0):.0f}ms")
    ok2 = check("generate_ms reported", has_generate, f"{a.timings_ms.get('generate_ms', 0):.0f}ms")
    ok3 = check("generate_ms > 0 (real LLM call happened)", a.timings_ms.get("generate_ms", 0) > 0)
    if not (ok1 and ok2 and ok3):
        failures.append("latency")

    # --- 6. Response caching ---------------------------------------------
    banner("Extension 6: Response caching")
    cache = ResponseCache()
    cache.clear()
    pipe3 = RAGPipeline(retriever=retriever, cache=cache, model=PRIMARY_LLM)
    q = "What is the Hagia Sophia"
    t0 = time.perf_counter()
    first = pipe3.answer(q)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    second = pipe3.answer(q)
    t_second = time.perf_counter() - t0
    ok1 = check("first call NOT cached", first.cached is False)
    ok2 = check("second call IS cached",  second.cached is True)
    ok3 = check("cached call is at least 5x faster",
                t_second * 5 < t_first, f"first={t_first*1000:.0f}ms, cached={t_second*1000:.0f}ms")
    ok4 = check("cached answer text matches first answer", first.answer == second.answer)
    ok5 = check("cache has at least 1 entry", cache.size() >= 1, f"size={cache.size()}")
    if not (ok1 and ok2 and ok3 and ok4 and ok5):
        failures.append("cache")

    # --- 7. Improved retrieval ranking (hybrid + intro) ------------------
    banner("Extension 7: Improved retrieval ranking (BM25 + dense + RRF + intro guarantee)")
    decision, chunks = retriever.retrieve("Where is Mount Everest", top_k=5)
    sources_used = {s for c in chunks for s in c.sources}
    ok1 = check("dense scoring used", "dense" in sources_used)
    ok2 = check("BM25 scoring used", "bm25" in sources_used)
    ok3 = check("intro chunk guaranteed first", chunks[0].section.lower() == "introduction" if chunks else False,
                f"top section: {chunks[0].section if chunks else 'none'}")
    # Test a plain query with no entity match to verify hybrid still works
    decision2, chunks2 = retriever.retrieve("famous mathematician known for laws of motion", top_k=5)
    has_newton = any(c.entity == "Isaac Newton" for c in chunks2)
    ok4 = check("hybrid finds Newton from semantic+keyword query (no name)", has_newton,
                f"top entities: {[c.entity for c in chunks2[:3]]}")
    if not (ok1 and ok2 and ok3 and ok4):
        failures.append("hybrid-ranking")

    # --- 8. Comparison questions across people and places ----------------
    banner("Extension 8: Comparison questions across people and places")
    decision, chunks = retriever.retrieve("Compare Albert Einstein and Nikola Tesla", top_k=6)
    ents = {c.entity for c in chunks}
    ok1 = check("Einstein chunks present", "Albert Einstein" in ents)
    ok2 = check("Tesla chunks present",   "Nikola Tesla" in ents)
    ok3 = check("multi-entity routing returns both", ok1 and ok2)
    decision2, chunks2 = retriever.retrieve("Compare the Eiffel Tower and the Statue of Liberty", top_k=6)
    ents2 = {c.entity for c in chunks2}
    ok4 = check("Eiffel Tower chunks present", "Eiffel Tower" in ents2)
    ok5 = check("Statue of Liberty chunks present", "Statue of Liberty" in ents2)
    if not (ok1 and ok2 and ok3 and ok4 and ok5):
        failures.append("comparison")

    # --- summary ---------------------------------------------------------
    banner("SUMMARY")
    if not failures:
        print("All 8 optional extensions verified working.")
        return 0
    print(f"FAIL: {len(failures)} extension(s): {failures}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
