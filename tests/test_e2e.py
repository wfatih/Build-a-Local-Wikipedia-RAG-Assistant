"""End-to-end smoke test for the Local Wikipedia RAG Assistant.

Runs every example query from the brief plus extra edge cases against the
live pipeline. Requires:
    * Ollama running locally with llama3.2:3b and nomic-embed-text pulled
    * data/rag.db already ingested

Outputs a summary table with PASS/FAIL/WARN per question and prints any
failures verbatim.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache.response_cache import ResponseCache
from src.generate.pipeline import RAGPipeline
from src.retrieve.retriever import Retriever


REFUSAL = "I don't know based on the provided context."
REFUSAL_RE = re.compile(r"don'?t know based on the provided context", re.IGNORECASE)


def must_contain(*needles: str):
    needles_l = [n.lower() for n in needles]
    def check(answer: str, _chunks):
        a = answer.lower()
        missing = [n for n in needles_l if n not in a]
        return (not missing, f"missing: {missing}" if missing else "")
    return check


def must_not_contain(*needles: str):
    needles_l = [n.lower() for n in needles]
    def check(answer: str, _chunks):
        a = answer.lower()
        present = [n for n in needles_l if n in a]
        return (not present, f"forbidden present: {present}" if present else "")
    return check


def must_refuse():
    def check(answer: str, _chunks):
        ok = bool(REFUSAL_RE.search(answer))
        return (ok, "" if ok else "did not refuse")
    return check


def must_route(target: str):
    def check_route(decision_target: str):
        return (decision_target == target, f"got route={decision_target}, expected {target}")
    return check_route


def must_have_entities(*entities: str):
    ents_l = {e.lower() for e in entities}
    def check(_answer, chunks):
        seen = {c.entity.lower() for c in chunks}
        missing = [e for e in ents_l if e not in seen]
        return (not missing, f"chunks missing entities: {missing}" if missing else "")
    return check


def all_checks(*checks):
    def run(answer, chunks):
        problems = []
        for c in checks:
            ok, msg = c(answer, chunks)
            if not ok:
                problems.append(msg)
        return (not problems, "; ".join(problems))
    return run


# -- test cases ---------------------------------------------------------------

CASES = [
    # (label, query, content_check, expected_route_or_None)
    ("people-1", "Who was Albert Einstein and what is he known for",
     all_checks(must_contain("relativity"), must_have_entities("Albert Einstein")),
     "person"),
    ("people-2", "What did Marie Curie discover",
     all_checks(must_contain("radium"), must_have_entities("Marie Curie")),
     "person"),
    ("people-3", "Why is Nikola Tesla famous",
     all_checks(must_contain("alternating current"), must_have_entities("Nikola Tesla")),
     "person"),
    ("people-4", "Compare Lionel Messi and Cristiano Ronaldo",
     all_checks(must_have_entities("Lionel Messi", "Cristiano Ronaldo")),
     "person"),
    ("people-5", "What is Frida Kahlo known for",
     all_checks(must_have_entities("Frida Kahlo")),
     "person"),

    ("places-1", "Where is the Eiffel Tower located",
     all_checks(must_contain("paris"), must_have_entities("Eiffel Tower")),
     "place"),
    ("places-2", "Why is the Great Wall of China important",
     all_checks(must_have_entities("Great Wall of China")),
     "place"),
    ("places-3", "What is Machu Picchu",
     all_checks(must_have_entities("Machu Picchu")),
     "place"),
    ("places-4", "What was the Colosseum used for",
     all_checks(must_contain("gladiator"), must_have_entities("Colosseum")),
     "place"),
    ("places-5", "Where is Mount Everest",
     all_checks(must_have_entities("Mount Everest")),
     "place"),

    ("mixed-1", "Which famous place is located in Turkey",
     all_checks(
         must_not_contain("eiffel tower is located in turkey",
                          "eiffel tower is in turkey",
                          "statue of liberty is in turkey",
                          "machu picchu is in turkey"),
     ),
     "place"),
    ("mixed-2", "Which person is associated with electricity",
     must_contain("tesla"),
     None),
    ("mixed-3", "Compare Albert Einstein and Nikola Tesla",
     all_checks(must_have_entities("Albert Einstein", "Nikola Tesla")),
     "person"),
    ("mixed-4", "Compare the Eiffel Tower and the Statue of Liberty",
     all_checks(must_have_entities("Eiffel Tower", "Statue of Liberty")),
     "place"),

    ("fail-1", "Who is the president of Mars", must_refuse(), None),
    ("fail-2", "Tell me about a random unknown person John Doe", must_refuse(), None),

    # extra edge cases — make sure custom names work
    ("extra-1", "Who is Kemal Sunal",
     all_checks(must_have_entities("Kemal Sunal")), "person"),
    ("extra-2", "Tell me about Sagopa Kajmer",
     all_checks(must_have_entities("Sagopa Kajmer")), "person"),
    ("extra-3", "What is the Hagia Sophia",
     all_checks(must_have_entities("Hagia Sophia")), "place"),
    ("extra-4", "Where is the Galata Tower",
     all_checks(must_contain("istanbul"), must_have_entities("Galata Tower")), "place"),
]


def main() -> int:
    print("Wiping response cache to force fresh generations…")
    ResponseCache().clear()

    retriever = Retriever()
    pipe = RAGPipeline(retriever=retriever, cache=None, model="llama3.2:3b")

    results = []
    for label, query, check, expected_route in CASES:
        t0 = time.time()
        try:
            ans = pipe.answer(query, top_k=5, history=None, use_cache=False)
        except Exception as e:  # noqa: BLE001
            results.append({
                "label": label, "query": query, "status": "ERROR",
                "msg": f"exception: {e}", "answer": "", "route": "",
                "entities": [], "elapsed": time.time() - t0,
            })
            continue
        chunks = ans.chunks
        entities = sorted({c.entity for c in chunks})
        ok, msg = check(ans.answer, chunks)
        route_ok = True
        route_msg = ""
        if expected_route is not None:
            route_ok = ans.routing.target == expected_route
            if not route_ok:
                route_msg = f"route={ans.routing.target} expected={expected_route}"
        status = "PASS" if (ok and route_ok) else ("FAIL" if not ok else "WARN")
        full_msg = "; ".join(m for m in (msg, route_msg) if m)
        results.append({
            "label": label, "query": query, "status": status, "msg": full_msg,
            "answer": ans.answer.strip(), "route": ans.routing.target,
            "entities": entities, "elapsed": time.time() - t0,
        })
        marker = {"PASS": "[OK]", "FAIL": "[FAIL]", "WARN": "[WARN]", "ERROR": "[ERR]"}[status]
        print(f"{marker:6s} {label:8s} ({ans.routing.target:7s}) {ans.timings_ms.get('retrieve_ms',0):6.0f}ms+{ans.timings_ms.get('generate_ms',0):6.0f}ms  {query[:55]}")
        if status != "PASS":
            print(f"           why: {full_msg}")
            print(f"           entities: {entities}")
            print(f"           answer: {ans.answer[:300]}")

    fails = [r for r in results if r["status"] in ("FAIL", "ERROR")]
    warns = [r for r in results if r["status"] == "WARN"]
    print()
    print(f"Total: {len(results)} | PASS: {len(results)-len(fails)-len(warns)} | "
          f"FAIL: {len(fails)} | WARN: {len(warns)}")
    out = ROOT / "tests" / "last_run.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detailed results: {out}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
