"""End-to-end RAG pipeline:
    user query -> route -> retrieve (dense+BM25 RRF) -> prompt build -> LLM
                                                                |
                                       optional cache lookup ---+
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Iterator

from src.config import PRIMARY_LLM, SELF_GROUNDING_DEFAULT, TOP_K
from src.cache.response_cache import ResponseCache
from src.generate.grounding import self_check
from src.generate.llm import chat, chat_stream
from src.log import log_event, new_request_id
from src.retrieve.retriever import RetrievedChunk, Retriever
from src.retrieve.router import RoutingDecision


REFUSAL = "I don't know based on the provided context."

# Phrases the model emits when it has no real answer but improvises instead of
# using the canonical refusal sentence. We rewrite these on the way out so the
# UI gets the exact required phrase from the brief.
_REFUSAL_PARAPHRASE_RE = __import__("re").compile(
    r"(?:there is no (?:information|mention|reference)|"
    r"the (?:provided )?context (?:does not|doesn't) (?:contain|mention|provide)|"
    r"i (?:cannot|can'?t) (?:find|determine|provide)|"
    r"no (?:information|details|data) (?:is|are) (?:provided|available|given)|"
    r"not (?:mentioned|provided|stated|covered) in the (?:provided )?context)",
    __import__("re").IGNORECASE,
)


def _normalise_refusal(answer: str) -> str:
    """If the model improvised a 'don't know' instead of the exact refusal
    sentence required by the brief, rewrite it. Only applies to short
    answers — substantive answers that happen to mention 'not mentioned'
    are left alone.
    """
    a = answer.strip()
    if len(a) > 240:
        return answer
    if REFUSAL.lower() in a.lower():
        return answer
    if _REFUSAL_PARAPHRASE_RE.search(a):
        return REFUSAL
    return answer


SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about famous people and "
    "famous places using ONLY the provided numbered context items. Follow these "
    "rules strictly:\n"
    "1. EVERY factual claim in your answer MUST be directly supported by one of "
    "the numbered context items. If you cannot point to a specific item that "
    "supports a claim, do not write it.\n"
    "2. If the context does not contain the answer, reply with EXACTLY this "
    "sentence and nothing else: \"I don't know based on the provided context.\"\n"
    "3. Never combine entities across context items in a way the context does "
    "not state. If item [1] is about the Eiffel Tower (Paris) and item [2] is "
    "about Ephesus (Turkey), do NOT write \"the Eiffel Tower is in Turkey\" — "
    "answer only with what each item actually says about the question.\n"
    "4. Do not invent facts, dates, names, or numbers, even if they sound "
    "plausible.\n"
    "5. EVERY factual sentence MUST end with a bracketed citation that "
    "matches the supporting context item, like [1] or [2]. If a sentence has "
    "no supporting item, do not write it. Do NOT echo the parenthetical "
    "chunk header (e.g. \"(person: X — Introduction)\") in your answer.\n"
    "6. Do NOT comment on items that don't help. Don't write things like "
    "\"[1] does not mention X\" or \"[2] discusses something else\". Just "
    "answer using the items that DO support what you say.\n"
    "7. Write a single coherent paragraph in flowing prose. Do not use "
    "bullet lists, do not write item-by-item commentary, do not include a "
    "preamble like \"Based on the provided context\".\n"
    "8. Keep answers concise (2-5 sentences) unless the user asks for detail.\n"
    "9. For comparison questions, your answer MUST mention each subject by "
    "name and state at least one fact about each that is supported by the "
    "context.\n"
)


_PRONOUN_RE = __import__("re").compile(
    r"\b(he|she|it|him|her|his|hers|its|they|their|them|"
    r"this|that|these|those)\b",
    __import__("re").IGNORECASE,
)
_FOLLOWUP_HINTS = (
    "tell me more", "more about", "what about", "and ", "also",
    "the same", "what else", "go deeper", "elaborate",
)


def _has_pronoun_or_followup(query: str) -> bool:
    q = query.lower()
    if _PRONOUN_RE.search(q):
        return True
    return any(h in q for h in _FOLLOWUP_HINTS)


def _augment_query_with_history(query: str, history: list[dict] | None, router) -> str:
    """For follow-up questions like "What year did *he* win", neither the
    keyword router nor cosine retrieval can resolve the pronoun. We patch
    that by harvesting any entity names mentioned in the last few turns and
    prefixing them to the retrieval query.

    Conditions for augmentation:
        1. There is recent history.
        2. The current query does NOT already name a known entity.
        3. EITHER the query contains a pronoun/follow-up cue, OR the most
           recent user turn referenced an entity (carry-over assumption).
    """
    if not history:
        return query
    decision = router.route(query)
    if decision.person_entities or decision.place_entities:
        return query

    pronoun_or_followup = _has_pronoun_or_followup(query)

    # Walk recent turns newest-first; the most recently mentioned entity is
    # the most likely referent.
    recent = list(reversed(history[-8:]))
    found: list[str] = []
    for m in recent:
        text = m.get("content", "")
        for ent in router.people + router.places:
            if ent.lower() in text.lower() and ent not in found:
                found.append(ent)
                if len(found) >= 2:
                    break
        if found:
            break  # only carry from the most recent referenced turn

    if not found:
        return query
    if not pronoun_or_followup:
        # If the user wrote a fully-formed question that doesn't reference
        # the previous topic, leave it alone. We only augment when the
        # current query *needs* the carry-over.
        return query
    return f"{', '.join(found)}. {query}"


def _build_prompt(query: str, chunks: list[RetrievedChunk], history: list[dict] | None) -> list[dict]:
    ctx_lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        ctx_lines.append(
            f"[{i}] ({c.type}: {c.title} — {c.section or 'overview'})\n{c.text}"
        )
    context_block = "\n\n".join(ctx_lines) if ctx_lines else "(no context retrieved)"

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])  # short conversational memory
    messages.append({
        "role": "user",
        "content": f"Question: {query}\n\nContext:\n{context_block}",
    })
    return messages


def _cache_key(query: str, chunk_ids: list[int], model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"|")
    h.update(query.strip().lower().encode("utf-8"))
    h.update(b"|")
    h.update(",".join(str(i) for i in sorted(chunk_ids)).encode())
    return h.hexdigest()


@dataclass
class RAGAnswer:
    query: str
    answer: str
    chunks: list[RetrievedChunk]
    routing: RoutingDecision
    model: str
    cached: bool = False
    timings_ms: dict = field(default_factory=dict)
    grounded: bool = False  # whether self-grounding pass ran
    grounding_dropped: int = 0  # how many sentences were pruned


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        cache: ResponseCache | None = None,
        model: str = PRIMARY_LLM,
        self_grounding: bool = SELF_GROUNDING_DEFAULT,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.cache = cache if cache is not None else ResponseCache()
        self.model = model
        self.self_grounding = self_grounding

    def answer(
        self,
        query: str,
        top_k: int = TOP_K,
        history: list[dict] | None = None,
        use_cache: bool = True,
    ) -> RAGAnswer:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        retrieval_query = _augment_query_with_history(query, history, self.retriever.router)
        decision, chunks = self.retriever.retrieve(retrieval_query, top_k=top_k)
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

        chunk_ids = [c.id for c in chunks]
        ck = _cache_key(query, chunk_ids, self.model)
        if use_cache and self.cache:
            cached = self.cache.get(ck)
            if cached is not None:
                timings["generate_ms"] = 0.0
                return RAGAnswer(
                    query=query, answer=cached, chunks=chunks, routing=decision,
                    model=self.model, cached=True, timings_ms=timings,
                )

        if not chunks:
            answer = "I don't know based on the provided context."
            timings["generate_ms"] = 0.0
            if use_cache and self.cache:
                self.cache.put(ck, answer)
            return RAGAnswer(
                query=query, answer=answer, chunks=chunks, routing=decision,
                model=self.model, cached=False, timings_ms=timings,
            )

        messages = _build_prompt(query, chunks, history)
        t1 = time.perf_counter()
        answer = chat(messages, model=self.model)
        answer = _normalise_refusal(answer)
        timings["generate_ms"] = (time.perf_counter() - t1) * 1000

        grounded = False
        dropped = 0
        if self.self_grounding and chunks:
            t2 = time.perf_counter()
            ctx = "\n\n".join(
                f"[{i+1}] ({c.type}: {c.title} — {c.section or 'overview'})\n{c.text}"
                for i, c in enumerate(chunks)
            )
            new = self_check(query, answer, ctx, self.model)
            timings["grounding_ms"] = (time.perf_counter() - t2) * 1000
            if new and new != answer:
                # Crude drop counter: difference in '.'-terminated chunks.
                dropped = max(0, answer.count(".") - new.count("."))
                answer = new
            grounded = True

        if use_cache and self.cache:
            self.cache.put(ck, answer)
        log_event(
            "answer",
            request_id=new_request_id(),
            model=self.model,
            query=query,
            route=decision.target,
            n_chunks=len(chunks),
            top_entity=chunks[0].entity if chunks else None,
            grounded=grounded,
            grounding_dropped=dropped,
            cached=False,
            **{k: round(v, 1) for k, v in timings.items()},
        )
        return RAGAnswer(
            query=query, answer=answer, chunks=chunks, routing=decision,
            model=self.model, cached=False, timings_ms=timings,
            grounded=grounded, grounding_dropped=dropped,
        )

    def answer_stream(
        self,
        query: str,
        top_k: int = TOP_K,
        history: list[dict] | None = None,
    ) -> tuple[RAGAnswer, Iterator[str]]:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        retrieval_query = _augment_query_with_history(query, history, self.retriever.router)
        decision, chunks = self.retriever.retrieve(retrieval_query, top_k=top_k)
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

        if not chunks:
            stub = RAGAnswer(query=query, answer="I don't know based on the provided context.",
                             chunks=[], routing=decision, model=self.model,
                             cached=False, timings_ms=timings)

            def _empty() -> Iterator[str]:
                yield stub.answer

            return stub, _empty()

        messages = _build_prompt(query, chunks, history)
        stub = RAGAnswer(query=query, answer="", chunks=chunks, routing=decision,
                         model=self.model, cached=False, timings_ms=timings)
        return stub, chat_stream(messages, model=self.model)
