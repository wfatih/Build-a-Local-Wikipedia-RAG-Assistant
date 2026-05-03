"""Startup health checks for the Streamlit UI."""
from __future__ import annotations

from dataclasses import dataclass

import requests

from src.config import (
    DB_PATH,
    EMBED_MODEL,
    OLLAMA_HOST,
    PRIMARY_LLM,
    SECONDARY_LLM,
)


@dataclass
class HealthCheck:
    name: str
    ok: bool
    detail: str
    fix_hint: str = ""


def check_ollama_running() -> HealthCheck:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
        return HealthCheck("Ollama daemon", True, "reachable at " + OLLAMA_HOST)
    except Exception as e:
        return HealthCheck(
            "Ollama daemon",
            False,
            f"cannot reach {OLLAMA_HOST}: {e}",
            "Start Ollama: open a terminal and run `ollama serve`.",
        )


def check_models() -> list[HealthCheck]:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
        installed = {m.get("name", "").split(":")[0]: m.get("name", "")
                     for m in r.json().get("models", [])}
        installed_full = {m.get("name", "") for m in r.json().get("models", [])}
    except Exception as e:
        return [HealthCheck("Models", False, f"cannot list models: {e}",
                            "Make sure Ollama is running.")]

    out: list[HealthCheck] = []
    for label, mid, required in (
        ("Primary LLM", PRIMARY_LLM, True),
        ("Embedding model", EMBED_MODEL, True),
        ("Secondary LLM (compare feature)", SECONDARY_LLM, False),
    ):
        base = mid.split(":")[0]
        present = mid in installed_full or base in installed
        if present:
            out.append(HealthCheck(label, True, f"{mid} pulled"))
        else:
            out.append(HealthCheck(
                label,
                False if required else True,
                f"{mid} not pulled",
                f"Run `ollama pull {mid}`."
                + ("" if required else " (optional — only used by 'compare two models')"),
            ))
    return out


def check_store_populated() -> HealthCheck:
    if not DB_PATH.exists():
        return HealthCheck(
            "Vector store",
            False,
            f"{DB_PATH} does not exist",
            "Run `python -m src.ingest.run_ingest` to build it.",
        )
    import sqlite3
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        people = conn.execute("SELECT COUNT(DISTINCT entity) FROM chunks WHERE type='person'").fetchone()[0]
        places = conn.execute("SELECT COUNT(DISTINCT entity) FROM chunks WHERE type='place'").fetchone()[0]
        conn.close()
    except Exception as e:
        return HealthCheck("Vector store", False, f"could not read store: {e}",
                           "Re-run ingestion: `python -m src.ingest.run_ingest --reset`.")
    if n < 100 or people < 20 or places < 20:
        return HealthCheck(
            "Vector store",
            False,
            f"only {n} chunks across {people} people + {places} places",
            "Run `python -m src.ingest.run_ingest` to ingest the full corpus.",
        )
    return HealthCheck(
        "Vector store",
        True,
        f"{n} chunks across {people} people + {places} places",
    )


def run_all() -> list[HealthCheck]:
    out = [check_ollama_running()]
    if out[0].ok:
        out.extend(check_models())
    out.append(check_store_populated())
    return out


def all_critical_ok(checks: list[HealthCheck]) -> bool:
    return all(c.ok for c in checks if "Secondary" not in c.name)
