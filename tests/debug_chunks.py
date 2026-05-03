"""Dump retrieved chunks for problematic queries so we can see what the LLM
saw vs what it produced."""
from __future__ import annotations
import io
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force UTF-8 stdout on Windows so we don't blow up on Turkish/Greek glyphs.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.retrieve.retriever import Retriever

QUERIES = [
    "Where is Mount Everest",
    "What is the Hagia Sophia",
    "Compare the Eiffel Tower and the Statue of Liberty",
    "Compare Albert Einstein and Nikola Tesla",
    "Tell me about Sagopa Kajmer",
    "What is Machu Picchu",
]

r = Retriever()
for q in QUERIES:
    print("\n" + "=" * 80)
    print("Q:", q)
    decision, chunks = r.retrieve(q, top_k=5)
    print(f"route={decision.target} ents={decision.person_entities + decision.place_entities}")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c.entity} — {c.section} (score={c.score:.3f}, src={','.join(c.sources)})")
        print(f"      {c.text[:220]}")
