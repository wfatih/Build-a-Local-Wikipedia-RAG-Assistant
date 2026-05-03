"""End-to-end ingestion entrypoint:
    1. Fetch Wikipedia articles for every entity in people.txt and places.txt
    2. Chunk them
    3. Embed each chunk via local Ollama
    4. Persist into the SQLite vector store
"""
from __future__ import annotations

import argparse
import time

from src.config import PEOPLE_FILE, PLACES_FILE
from src.ingest.wikipedia import fetch_all, read_entity_list
from src.chunk.chunker import chunk_doc
from src.embed.embedder import embed_texts
from src.store.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia data into the local RAG store.")
    parser.add_argument("--force-fetch", action="store_true", help="Bypass the on-disk Wikipedia cache.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the vector store before ingesting.")
    parser.add_argument("--only", choices=["people", "places"], help="Restrict ingestion to a single type.")
    args = parser.parse_args()

    store = VectorStore()
    if args.reset:
        store.reset()
    store.init_schema()

    plan: list[tuple[str, list[str]]] = []
    if args.only != "places":
        plan.append(("person", read_entity_list(PEOPLE_FILE)))
    if args.only != "people":
        plan.append(("place", read_entity_list(PLACES_FILE)))

    t0 = time.time()
    for type_, entities in plan:
        print(f"\n=== Fetching {len(entities)} {type_}s ===")
        docs = fetch_all(entities, type_, force=args.force_fetch)
        print(f"\n=== Chunking & embedding {type_}s ===")
        for doc in docs:
            chunks = chunk_doc(doc)
            if not chunks:
                print(f"  [SKIP] {doc.entity}: no chunks")
                continue
            existing = store.count_for_doc(doc.entity, type_)
            if existing == len(chunks) and not args.reset:
                print(f"  [CACHED] {doc.entity}: {existing} chunks already embedded")
                continue
            store.delete_doc(doc.entity, type_)
            texts = [c["text"] for c in chunks]
            vecs = embed_texts(texts)
            store.add_chunks(doc, chunks, vecs)
            print(f"  [EMBEDDED] {doc.entity}: {len(chunks)} chunks")

    dt = time.time() - t0
    total = store.total_chunks()
    print(f"\nDone. {total} chunks total in {dt:.1f}s.")


if __name__ == "__main__":
    main()
