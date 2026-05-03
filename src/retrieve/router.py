"""Lightweight rule-based query router: decides whether the query is about a
person, a place, or both, using entity-name matching + keyword cues.

Returns one of: "person", "place", "mixed", or "unknown".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.config import PEOPLE_FILE, PLACES_FILE
from src.ingest.wikipedia import read_entity_list


PERSON_KEYWORDS = {
    "who", "person", "scientist", "writer", "artist", "actor", "actress",
    "singer", "musician", "footballer", "athlete", "leader", "president",
    "queen", "king", "philosopher", "physicist", "mathematician", "rapper",
    "born", "died", "biography", "career", "discovered", "invented",
}
PLACE_KEYWORDS = {
    "where", "place", "location", "country", "city", "monument", "tower",
    "wall", "wonder", "mountain", "river", "temple", "palace", "museum",
    "located", "built", "stands", "situated", "landmark", "bridge",
}


@dataclass
class RoutingDecision:
    target: str  # "person" | "place" | "mixed" | "unknown"
    person_entities: list[str]
    place_entities: list[str]
    rationale: str


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Bounded Levenshtein. Returns `cap+1` if min distance exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            best = min(best, cur[j])
        if best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


_GENERIC_TOKENS = {"city", "tower", "wall", "square", "falls", "palace",
                   "mosque", "the", "of", "great", "mount", "saint", "san"}


def _token_is_unique(token: str, freq: dict[str, int]) -> bool:
    return (
        len(token) >= 4
        and freq.get(token, 0) == 1
        and token not in _GENERIC_TOKENS
    )


def _find_entities(query: str, entities: list[str]) -> list[str]:
    qn = _normalise(query)
    last_freq: dict[str, int] = {}
    first_freq: dict[str, int] = {}
    for e in entities:
        toks = _normalise(e).split()
        last_freq[toks[-1]] = last_freq.get(toks[-1], 0) + 1
        first_freq[toks[0]] = first_freq.get(toks[0], 0) + 1

    query_tokens = re.findall(r"[a-zçğıöşü]+", qn)

    hits: list[str] = []
    for e in entities:
        en = _normalise(e)
        toks = en.split()
        # 1. Whole-name exact match.
        if re.search(rf"\b{re.escape(en)}\b", qn):
            hits.append(e)
            continue
        # 2. Exact match on a uniquely-owned distinctive token.
        candidates = []
        if _token_is_unique(toks[-1], last_freq):
            candidates.append(toks[-1])
        if len(toks) > 1 and _token_is_unique(toks[0], first_freq):
            candidates.append(toks[0])
        matched = False
        for cand in candidates:
            if re.search(rf"\b{re.escape(cand)}\b", qn):
                hits.append(e)
                matched = True
                break
        if matched:
            continue
        # 3. Fuzzy match: typo-tolerant. Match a distinctive token against
        # any query token, allowing 1-2 edits depending on length.
        for cand in candidates:
            cap = 1 if len(cand) <= 7 else 2
            done = False
            for qt in query_tokens:
                if qt == cand:
                    continue
                if len(qt) < max(4, len(cand) - cap):
                    continue
                if abs(len(qt) - len(cand)) > cap:
                    continue
                if _edit_distance(qt, cand, cap=cap) <= cap:
                    hits.append(e)
                    done = True
                    break
            if done:
                break
    return hits


class Router:
    def __init__(
        self,
        people_file: Path = PEOPLE_FILE,
        places_file: Path = PLACES_FILE,
    ) -> None:
        self.people = read_entity_list(people_file)
        self.places = read_entity_list(places_file)

    def route(self, query: str) -> RoutingDecision:
        qn = _normalise(query)
        person_hits = _find_entities(query, self.people)
        place_hits = _find_entities(query, self.places)

        if person_hits and place_hits:
            return RoutingDecision("mixed", person_hits, place_hits,
                                   f"matched person(s) {person_hits} and place(s) {place_hits}")
        if person_hits:
            return RoutingDecision("person", person_hits, [],
                                   f"matched person(s) {person_hits}")
        if place_hits:
            return RoutingDecision("place", [], place_hits,
                                   f"matched place(s) {place_hits}")

        tokens = set(re.findall(r"[a-zçğıöşü]+", qn))
        person_score = len(tokens & PERSON_KEYWORDS)
        place_score = len(tokens & PLACE_KEYWORDS)
        if person_score and place_score:
            return RoutingDecision("mixed", [], [], "person+place keyword overlap")
        if person_score > place_score:
            return RoutingDecision("person", [], [], "person keywords dominate")
        if place_score > person_score:
            return RoutingDecision("place", [], [], "place keywords dominate")
        return RoutingDecision("unknown", [], [], "no entity or keyword signal")
