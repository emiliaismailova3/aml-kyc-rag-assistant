"""SQLite-backed request logging for the /ask API endpoint.

Every call to POST /ask is logged with its question, answer, retrieved
sources, latency, and whether it went through the agent or the plain RAG
pipeline. Kept as a tiny, dependency-free SQLite table rather than a full
logging stack, since that's proportionate to a portfolio-project API.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "logs" / "requests.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    latency_ms REAL NOT NULL
);
"""


@contextmanager
def _connection(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_request(
    question: str,
    answer: str,
    sources: list[dict],
    pipeline: str,
    latency_ms: float,
    db_path: Path = DB_PATH,
) -> None:
    with _connection(db_path) as conn:
        conn.execute(
            "INSERT INTO request_log (timestamp, question, answer, sources_json, pipeline, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                answer,
                json.dumps(sources, ensure_ascii=False),
                pipeline,
                latency_ms,
            ),
        )


def fetch_recent_requests(limit: int = 50, db_path: Path = DB_PATH) -> list[dict]:
    with _connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM request_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
