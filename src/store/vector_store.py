"""Hand-rolled vector store: SQLite for metadata + numpy for similarity.

Why not Chroma? The brief asks us to favour native functionality over libraries
that do the core work for us. Storing float32 embeddings as a SQLite BLOB and
loading them into a single numpy matrix at query time gives us:
    - one process, no extra service
    - O(N*d) cosine via a single matmul (fast for our N ~ a few thousand)
    - exact, deterministic results — easy to reason about
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    entity    TEXT    NOT NULL,
    title     TEXT    NOT NULL,
    type      TEXT    NOT NULL CHECK (type IN ('person', 'place')),
    url       TEXT    NOT NULL,
    section   TEXT,
    position  INTEGER NOT NULL,
    tokens    INTEGER NOT NULL,
    text      TEXT    NOT NULL,
    embedding BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_entity ON chunks(entity);
CREATE INDEX IF NOT EXISTS idx_chunks_type   ON chunks(type);
"""


class VectorStore:
    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._cache: dict | None = None  # in-memory matrix cache

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False: Streamlit reruns the script on different
            # worker threads. The store is single-writer in this app, so this
            # is safe.
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init_schema(self) -> None:
        c = self._connect()
        c.executescript(SCHEMA)
        c.commit()

    def reset(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self.db_path.exists():
            self.db_path.unlink()
        self._cache = None

    def total_chunks(self) -> int:
        return self._connect().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def count_for_doc(self, entity: str, type_: str) -> int:
        return self._connect().execute(
            "SELECT COUNT(*) FROM chunks WHERE entity=? AND type=?",
            (entity, type_),
        ).fetchone()[0]

    def delete_doc(self, entity: str, type_: str) -> None:
        c = self._connect()
        c.execute("DELETE FROM chunks WHERE entity=? AND type=?", (entity, type_))
        c.commit()
        self._cache = None

    def add_chunks(self, doc, chunks: Iterable[dict], vectors: np.ndarray) -> None:
        c = self._connect()
        rows = []
        for i, ch in enumerate(chunks):
            v = vectors[i].astype(np.float32, copy=False).tobytes()
            rows.append((
                doc.entity,
                doc.title,
                doc.type,
                doc.url,
                ch["section"],
                ch["position"],
                ch["tokens"],
                ch["text"],
                v,
            ))
        c.executemany(
            "INSERT INTO chunks (entity,title,type,url,section,position,tokens,text,embedding) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        c.commit()
        self._cache = None

    def _load_matrix(self, type_filter: str | None) -> dict:
        c = self._connect()
        if type_filter is None:
            cur = c.execute("SELECT id,entity,title,type,url,section,position,text,embedding FROM chunks")
        else:
            cur = c.execute(
                "SELECT id,entity,title,type,url,section,position,text,embedding FROM chunks WHERE type=?",
                (type_filter,),
            )
        ids: list[int] = []
        meta: list[dict] = []
        vecs: list[np.ndarray] = []
        for row in cur:
            ids.append(row["id"])
            meta.append({
                "id": row["id"],
                "entity": row["entity"],
                "title": row["title"],
                "type": row["type"],
                "url": row["url"],
                "section": row["section"],
                "position": row["position"],
                "text": row["text"],
            })
            vecs.append(np.frombuffer(row["embedding"], dtype=np.float32))
        if not vecs:
            return {"ids": [], "meta": [], "matrix": np.zeros((0, 1), dtype=np.float32)}
        matrix = np.vstack(vecs)
        return {"ids": ids, "meta": meta, "matrix": matrix}

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        type_filter: str | None = None,
        entity_filter: list[str] | None = None,
    ) -> list[dict]:
        cache_key = type_filter or "__all__"
        if self._cache is None or cache_key not in self._cache:
            if self._cache is None:
                self._cache = {}
            self._cache[cache_key] = self._load_matrix(type_filter)
        bundle = self._cache[cache_key]
        M = bundle["matrix"]
        if M.shape[0] == 0:
            return []
        scores = M @ query_vec.astype(np.float32)
        if entity_filter:
            allow = set(entity_filter)
            mask = np.array([m["entity"] in allow for m in bundle["meta"]], dtype=bool)
            if mask.any():
                # Push disallowed rows to -inf so they never crack top-k.
                scores = np.where(mask, scores, -np.inf)
        k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        out = []
        for i in idx:
            if not np.isfinite(scores[i]):
                continue
            m = dict(bundle["meta"][i])
            m["score"] = float(scores[i])
            out.append(m)
        return out

    def all_for_bm25(
        self,
        type_filter: str | None = None,
        entity_filter: list[str] | None = None,
    ) -> list[dict]:
        c = self._connect()
        sql = "SELECT id,entity,title,type,url,section,position,text FROM chunks"
        clauses: list[str] = []
        params: list = []
        if type_filter is not None:
            clauses.append("type=?")
            params.append(type_filter)
        if entity_filter:
            placeholders = ",".join("?" * len(entity_filter))
            clauses.append(f"entity IN ({placeholders})")
            params.extend(entity_filter)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [dict(r) for r in c.execute(sql, params)]

    def get_intro_chunks(self, entity: str, type_: str, max_chunks: int = 1) -> list[dict]:
        """Return the first `max_chunks` chunks of an entity's article, in
        order. The first chunk(s) are almost always the lead/Introduction —
        the highest-information passage for "who is X" / "where is X" style
        questions.
        """
        c = self._connect()
        cur = c.execute(
            "SELECT id,entity,title,type,url,section,position,text "
            "FROM chunks WHERE entity=? AND type=? ORDER BY position ASC LIMIT ?",
            (entity, type_, max_chunks),
        )
        return [dict(r) for r in cur]

    def stats(self) -> dict:
        c = self._connect()
        total = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        people = c.execute("SELECT COUNT(DISTINCT entity) FROM chunks WHERE type='person'").fetchone()[0]
        places = c.execute("SELECT COUNT(DISTINCT entity) FROM chunks WHERE type='place'").fetchone()[0]
        return {"chunks": total, "people": people, "places": places}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
