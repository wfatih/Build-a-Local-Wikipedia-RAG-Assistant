"""Tiny SQLite-backed response cache. Key is sha256(query|model|chunk_ids)."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from src.config import CACHE_DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key       TEXT PRIMARY KEY,
    answer    TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class ResponseCache:
    def __init__(self, db_path: Path = CACHE_DB_PATH) -> None:
        self.db_path = Path(db_path)
        # check_same_thread=False: Streamlit reruns hit different threads; this
        # cache is small and used through a single connection in-process.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT answer FROM responses WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, answer: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, answer, created_at) VALUES (?,?,?)",
            (key, answer, time.time()),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM responses")
        self._conn.commit()

    def size(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
