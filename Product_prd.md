# Product Requirements Document — Local Wikipedia RAG Assistant

**Author:** Fatih Çakır (150220086)
**Course:** ITU BLG483E — Artificial Intelligence Aided Computer Engineering
**Project:** 3 — Build a Local Wikipedia RAG Assistant

---

## 1. Background

Project 1 produced a search-and-retrieve system. Project 2 explored AI-driven workflows. This project combines both: a complete RAG (Retrieval-Augmented Generation) application that ingests Wikipedia data, indexes it locally, and answers natural-language questions using a locally hosted language model. The system must run end-to-end on the student's laptop — no remote LLM service, no remote vector database.

## 2. Goals

- Provide a ChatGPT-style chat interface that answers questions about a fixed set of famous people and famous places.
- Ground every generated answer in retrieved Wikipedia text so factual claims are auditable.
- Return a refusal sentence (`"I don't know based on the provided context."`) when the corpus does not contain the answer.
- Be reproducible by a third party using only the README.

## 3. Non-goals

- General-purpose chat outside the people/places domain.
- Live Wikipedia fetching at query time. Ingestion is offline; queries are served from the local store.
- Multi-user / multi-tenant deployment.

## 4. Personas

- **The instructor** — runs `git clone`, follows the README, expects a working chat in under five minutes (excluding model downloads).
- **A student exploring RAG** — reads the source to understand each stage; expects the core mechanics (chunking, similarity, BM25) to be implemented from first principles, not hidden in a library.

## 5. Functional requirements

### 5.1 Ingestion

- The system MUST ingest at least 20 famous people and 20 famous places. The shipped configuration ingests **30 + 30**.
- The minimum entity set defined by the brief MUST be present: 10 people and 10 places listed verbatim in `data/people.txt` and `data/places.txt`.
- Ingestion MUST be idempotent — re-running without `--reset` must not duplicate chunks.
- Ingestion MUST cache fetched Wikipedia text on disk so subsequent runs are offline.

### 5.2 Chunking

- Chunks MUST preserve paragraph boundaries when possible.
- Each chunk MUST carry: source entity, type (`person` or `place`), section heading, position, URL.
- A single oversize paragraph MUST fall back to sentence-level packing.

### 5.3 Embedding & storage

- Embeddings MUST be produced by a local model (no external API). The shipped configuration uses `nomic-embed-text` over Ollama.
- The vector store MUST run locally. The shipped implementation uses SQLite + NumPy with hand-rolled cosine similarity.
- The store MUST support metadata filtering by `type` so the retriever can target only people, only places, or both.

### 5.4 Retrieval

- The system MUST classify each query into `person`, `place`, `mixed`, or `unknown`.
- For `mixed` queries it MUST retrieve from both type slices and merge.
- Retrieval MUST combine dense and lexical signals (dense + BM25 with Reciprocal Rank Fusion).

### 5.5 Generation

- The LLM MUST run locally via Ollama. The shipped default is `llama3.2:3b`.
- Generation MUST be conditioned on the retrieved context only.
- The output MUST cite the context items it relies on inline using `[1] [2]` markers.
- When no context is retrieved or the context is insufficient, the model MUST emit the refusal sentence exactly.

### 5.6 Chat interface

- A user MUST be able to: ask a question, see the answer, optionally view retrieved context, and clear/reset the session.
- A web UI (Streamlit) and a CLI MUST both be provided.

### 5.7 Optional extensions (all delivered)

- Streaming responses, source citations, chat memory, dual-model comparison, per-stage latency measurement, response caching, hybrid retrieval ranking, comparison questions across people and places.

## 6. Non-functional requirements

- **Privacy:** no data leaves `localhost` after ingestion is complete.
- **Performance:** typical end-to-end latency on a modern laptop with `llama3.2:3b` should sit under ~6 seconds for a routine question; cached responses must return in <50ms.
- **Reproducibility:** the README + `requirements.txt` must be sufficient — no hidden environment variables, no per-machine paths.

## 7. Acceptance criteria

A reviewer SHOULD be able to:

1. Clone the repository, install Python deps, install Ollama, pull the two required models.
2. Run `python -m src.ingest.run_ingest` and observe ≥60 successful entity ingestions.
3. Launch `streamlit run src/ui/streamlit_app.py` and ask each example question from the brief.
4. Observe correct routing, citations, and refusals on the failure-case prompts.

## 8. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Wikipedia rate-limits ingestion | Custom `User-Agent`, retry with exponential back-off, on-disk cache. |
| Model hallucinates beyond the corpus | Strict system prompt; empty-context short-circuit; cited generation. |
| Embeddings or LLM unavailable | Setup script + clear error messages; CLI surfaces `ollama` reachability. |
| Vector store grows unboundedly | Out of scope for this corpus (60 entities ≈ a few thousand chunks); production guidance lives in `recommendation.md`. |

## 9. Out of scope

- Authentication / user accounts
- Multilingual queries (Wikipedia source is English; query understanding works on English keywords)
- Mobile clients
- Online learning / fine-tuning
