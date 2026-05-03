# Production Deployment Recommendations

The shipped system is a single-process, single-user, laptop-grade RAG assistant. The design is deliberately spartan so that every layer is auditable. This document records what we would change, and why, to take it to production.

---

## 1. Topology

| Layer | Local prototype | Production target |
| --- | --- | --- |
| LLM runtime | Ollama on the laptop | A managed inference cluster (vLLM / TGI / Ray Serve) on GPUs, fronted by a small autoscaler. Keep the same model contract so client code is unchanged. |
| Embedding service | Ollama `nomic-embed-text` | A dedicated embedding service (same model or `bge-large-en-v1.5`), batched, separate from the generation service. |
| Vector store | SQLite + NumPy in-process | Managed approximate-NN (Qdrant, Weaviate, pgvector + IVFFlat/HNSW). Switch on dataset size; below ~1M chunks the SQLite design is still workable. |
| Metadata DB | Same SQLite | Postgres. Reuse the same schema. |
| API | None (in-process) | A thin FastAPI service exposing `POST /chat`, `POST /ingest`, `GET /healthz`. |
| UI | Streamlit | Either a hardened Streamlit deployment or a custom Next.js front-end depending on UX needs. |
| Cache | SQLite KV | Redis (TTL + LRU). |
| Auth | None | OIDC, per-tenant API keys. |

## 2. Ingestion

- Move from "for-loop fetcher" to a Celery / Temporal pipeline with idempotent steps and retries.
- Track entity-level lineage: which articles, which revision id, which embedding model version. When the embedding model changes, embeddings must be re-computed; metadata makes this an explicit migration.
- Schedule a periodic re-ingestion job (daily/weekly) so Wikipedia edits flow in.
- Add content de-duplication and language detection if the corpus expands beyond English.

## 3. Retrieval quality

- Replace exact cosine over a NumPy matrix with HNSW (`hnswlib`, Qdrant). Keep the dense + BM25 + RRF approach — it generalises.
- Add a re-ranker (cross-encoder, e.g. `BAAI/bge-reranker-base`) over the top-50 RRF candidates. Adds latency but is the single highest-impact retrieval upgrade.
- Add query expansion (LLM-rewriter or HyDE) for short, ambiguous queries.
- Track retrieval quality with an offline judging set: every prompt has a gold passage; run nDCG@k weekly.

## 4. Generation quality and safety

- Add answer self-check: after generation, ask the LLM to mark each sentence as "supported / unsupported / partially supported" against the retrieved context, and rewrite or refuse accordingly.
- Surface explicit citations as clickable links to the original Wikipedia revision id.
- Persist prompts + answers + retrieved chunk ids for audit and offline evaluation. Mask any PII before storage.

## 5. Observability

- Stage-level metrics: `retrieve_latency`, `generate_latency`, `tokens_in`, `tokens_out`, `cache_hit_rate`, `refusal_rate`, `chunk_score_distribution`.
- Per-request structured logs keyed by `request_id`.
- LLM evaluation dashboard fed by a nightly run over a fixed prompt set; alert on regressions.

## 6. Cost & performance

- Cache aggressively at the **retrieval** layer (query → ranked chunk-ids), in addition to the response cache. Retrieval is repeatable; generation is the expensive part but small variance in retrieval defeats the response cache, so add a normalised-query layer.
- Quantise the LLM (Q4_K_M or AWQ) on-prem; for a managed deployment, prefer a small instruction-tuned model and use the re-ranker to push quality.
- Stream tokens to the client; first-token latency dominates user perception.

## 7. Security

- Treat retrieved context as untrusted input — Wikipedia is public, but a future ingestion source might not be. Sandbox the rendered HTML, escape output paths.
- Rate-limit per tenant.
- Strip secrets from the prompt and the cache key.

## 8. Failure modes worth pre-engineering

- **Embedding/LLM unreachable:** the pipeline currently raises; in production return a structured 503 and surface a "service unavailable" UI state, not a silent retry storm.
- **Vector drift:** if you change the embedding model, mark old embeddings stale and refuse to mix dimensions.
- **Hot entities:** popular people/places will dominate cache and retrieval. Add a per-entity QPS guardrail.

## 9. Migration path from this prototype

1. Wrap `RAGPipeline` with a FastAPI handler. The current dataclasses (`RAGAnswer`, `RetrievedChunk`) translate to JSON unchanged.
2. Swap `VectorStore` for a Qdrant-backed implementation behind the same `search` / `add_chunks` interface — the rest of the code is agnostic.
3. Move `ResponseCache` from SQLite to Redis behind the same `get` / `put` API.
4. Containerise with two images: `app` (FastAPI + retriever code) and `ingester` (the same code base, different entry point).
5. Add CI: `pytest` over a fixed mini-corpus; an offline retrieval-quality job using the gold set.

The core abstractions — `Retriever`, `RAGPipeline`, `Router`, `BM25`, `VectorStore`, `ResponseCache` — were sized so each one is replaceable in isolation. None of the production changes above require redesigning the others.
