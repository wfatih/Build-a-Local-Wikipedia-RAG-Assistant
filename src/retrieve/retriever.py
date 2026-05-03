"""Hybrid retriever: dense (cosine over Ollama embeddings) + lexical (BM25),
combined with Reciprocal Rank Fusion. Routing decides which slice of the corpus
to search; for `mixed` queries we run two routed sub-retrievals and merge.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import RERANK_CANDIDATE_POOL, RRF_K, TOP_K, USE_RERANKER_DEFAULT
from src.embed.embedder import embed_query
from src.retrieve.bm25 import BM25
from src.retrieve import reranker as _reranker
from src.retrieve.router import Router, RoutingDecision
from src.store.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    id: int
    entity: str
    title: str
    type: str
    url: str
    section: str
    text: str
    score: float
    sources: list[str]  # ["dense", "bm25"]


def _rrf(rankings: list[list[dict]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion across multiple ranked lists keyed by chunk id."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            cid = item["id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return fused


class Retriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        router: Router | None = None,
        use_reranker: bool = USE_RERANKER_DEFAULT,
    ) -> None:
        self.store = store or VectorStore()
        self.store.init_schema()
        self.router = router or Router()
        self.use_reranker = use_reranker
        self._bm25_cache: dict[str, BM25] = {}

    def _bm25_for(self, type_filter: str | None, entity_filter: list[str] | None = None) -> BM25:
        ent_key = ",".join(sorted(entity_filter)) if entity_filter else ""
        key = f"{type_filter or '__all__'}|{ent_key}"
        if key not in self._bm25_cache:
            docs = self.store.all_for_bm25(type_filter, entity_filter=entity_filter)
            self._bm25_cache[key] = BM25(docs)
        return self._bm25_cache[key]

    def _intro_chunks_for(
        self, entities: list[str], type_filter: str | None,
    ) -> list[dict]:
        """Force-include each entity's lead chunk (position=0). Wikipedia
        leads carry the canonical "who/what/where" answer; a pure cosine
        retrieval often ranks them below trivia like "External links".
        """
        if not entities or type_filter is None:
            return []
        out: list[dict] = []
        for ent in entities:
            for c in self.store.get_intro_chunks(ent, type_filter, max_chunks=1):
                out.append(c)
        return out

    def _retrieve_slice(
        self,
        query: str,
        type_filter: str | None,
        top_k: int,
        entity_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        qv = embed_query(query)
        # When the cross-encoder reranker is on, pull a deeper candidate
        # pool so the reranker has something to choose from.
        pool_mult = (RERANK_CANDIDATE_POOL // max(1, top_k)) if (
            self.use_reranker and _reranker.is_available()
        ) else 3
        dense_hits = self.store.search(
            qv, top_k=top_k * pool_mult, type_filter=type_filter, entity_filter=entity_filter,
        )
        bm25 = self._bm25_for(type_filter, entity_filter=entity_filter)
        bm25_hits_raw = bm25.top_k(query, top_k * pool_mult)
        bm25_hits = [d for d, _ in bm25_hits_raw]
        intro_hits = self._intro_chunks_for(entity_filter or [], type_filter)

        fused = _rrf([dense_hits, bm25_hits])

        meta_by_id: dict[int, dict] = {}
        for d in dense_hits:
            meta_by_id.setdefault(d["id"], d)
        for d in bm25_hits:
            meta_by_id.setdefault(d["id"], d)
        for d in intro_hits:
            meta_by_id.setdefault(d["id"], d)

        dense_ids = {d["id"] for d in dense_hits}
        bm25_ids = {d["id"] for d in bm25_hits}
        intro_ids = {d["id"] for d in intro_hits}

        # Reserve one slot per intro chunk; let RRF compete for the rest.
        out: list[RetrievedChunk] = []
        used: set[int] = set()
        for d in intro_hits:
            cid = d["id"]
            sources = ["intro"]
            if cid in dense_ids:
                sources.append("dense")
            if cid in bm25_ids:
                sources.append("bm25")
            out.append(RetrievedChunk(
                id=cid,
                entity=d["entity"],
                title=d["title"],
                type=d["type"],
                url=d["url"],
                section=d.get("section") or "",
                text=d["text"],
                # Give intro chunks a high synthetic score so any downstream
                # display sorts them at the top.
                score=1.0,
                sources=sources,
            ))
            used.add(cid)

        remaining = max(0, top_k - len(out))
        if remaining and fused:
            ranked = sorted(fused.items(), key=lambda x: -x[1])
            # Optional cross-encoder reranking. Build the candidate pool from
            # the top-N RRF entries (not yet used by intro reservation) and
            # let the reranker pick the best `remaining`.
            if self.use_reranker and _reranker.is_available():
                pool_dicts: list[dict] = []
                for cid, _score in ranked[:RERANK_CANDIDATE_POOL]:
                    if cid in used:
                        continue
                    m = dict(meta_by_id[cid])
                    m["_rrf"] = _score
                    pool_dicts.append(m)
                reranked = _reranker.rerank(query, pool_dicts, remaining)
                for m in reranked:
                    cid = m["id"]
                    sources = []
                    if cid in dense_ids:
                        sources.append("dense")
                    if cid in bm25_ids:
                        sources.append("bm25")
                    sources.append("rerank")
                    out.append(RetrievedChunk(
                        id=cid,
                        entity=m["entity"],
                        title=m["title"],
                        type=m["type"],
                        url=m["url"],
                        section=m.get("section") or "",
                        text=m["text"],
                        score=float(m.get("rerank_score", m.get("_rrf", 0.0))),
                        sources=sources,
                    ))
                    used.add(cid)
                    if len(out) >= top_k:
                        break
            else:
                for cid, score in ranked:
                    if cid in used:
                        continue
                    m = meta_by_id[cid]
                    sources = []
                    if cid in dense_ids:
                        sources.append("dense")
                    if cid in bm25_ids:
                        sources.append("bm25")
                    out.append(RetrievedChunk(
                        id=cid,
                        entity=m["entity"],
                        title=m["title"],
                        type=m["type"],
                        url=m["url"],
                        section=m.get("section") or "",
                        text=m["text"],
                        score=score,
                        sources=sources,
                    ))
                    used.add(cid)
                    if len(out) >= top_k:
                        break
        return out

    def _retrieve_multi_entity(
        self,
        query: str,
        type_filter: str,
        entities: list[str],
        total_k: int,
    ) -> list[RetrievedChunk]:
        """Per-entity retrieval — guarantees each named entity contributes
        chunks. Critical for "Compare X and Y" questions where a joint top-K
        can otherwise be dominated by a single entity.
        """
        if not entities:
            return self._retrieve_slice(query, type_filter, total_k)
        per = max(2, total_k // len(entities))
        out: list[RetrievedChunk] = []
        seen: set[int] = set()
        for ent in entities:
            for c in self._retrieve_slice(query, type_filter, per, entity_filter=[ent]):
                if c.id in seen:
                    continue
                seen.add(c.id)
                out.append(c)
        return out

    def retrieve(self, query: str, top_k: int = TOP_K) -> tuple[RoutingDecision, list[RetrievedChunk]]:
        decision = self.router.route(query)
        if decision.target == "mixed":
            half = max(2, top_k // 2)
            a = self._retrieve_multi_entity(query, "person", decision.person_entities, half)
            b = self._retrieve_multi_entity(query, "place", decision.place_entities, top_k - half)
            return decision, a + b
        if decision.target == "person":
            if len(decision.person_entities) >= 2:
                return decision, self._retrieve_multi_entity(
                    query, "person", decision.person_entities, top_k,
                )
            return decision, self._retrieve_slice(
                query, "person", top_k,
                entity_filter=decision.person_entities or None,
            )
        if decision.target == "place":
            if len(decision.place_entities) >= 2:
                return decision, self._retrieve_multi_entity(
                    query, "place", decision.place_entities, top_k,
                )
            return decision, self._retrieve_slice(
                query, "place", top_k,
                entity_filter=decision.place_entities or None,
            )
        # Unknown: search the entire corpus, no entity constraint.
        return decision, self._retrieve_slice(query, None, top_k)
