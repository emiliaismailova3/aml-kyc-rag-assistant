"""Relational database layer (SQLAlchemy Core): companies, invoices, logs.

Works on SQLite (default, no setup) and PostgreSQL (set DATABASE_URL). Only
the document-chunk table with its vector column is PostgreSQL-specific; that
lives in src/pgvector_store.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine

from src.config import PROJECT_ROOT, get_database_url

logger = logging.getLogger(__name__)

metadata = MetaData()

companies = Table(
    "companies",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(200), nullable=False),
    Column("voen", String(10), nullable=False, unique=True),
)

invoices = Table(
    "invoices",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("file_name", String(300)),
    Column("company_id", Integer, ForeignKey("companies.id"), nullable=True),
    Column("company_name", String(200)),
    Column("voen", String(10)),
    Column("invoice_number", String(100)),
    Column("issue_date", Date),
    Column("total_amount", Numeric(14, 2)),
    Column("currency", String(3)),
    Column("needs_human_review", Boolean, nullable=False, default=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# Question/answer log (the SQLite request_log in logging_db.py is the default
# store; this table is used instead when DATABASE_URL points to PostgreSQL).
request_logs = Table(
    "request_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("sources_json", Text, nullable=False),
    Column("pipeline", String(20), nullable=False),
    Column("latency_ms", Float, nullable=False),
)

# One row per LLM call attempt (written by src/llm_client.py). No prompt text or
# personal data is stored: only sizes, timing, cost and status.
llm_calls = Table(
    "llm_calls",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("provider", String(30), nullable=False),
    Column("model", String(100), nullable=False),
    Column("task", String(50)),
    Column("tier", String(10)),
    Column("latency_ms", Float, nullable=False),
    Column("input_tokens", Integer, nullable=False, default=0),
    Column("output_tokens", Integer, nullable=False, default=0),
    Column("cost_usd", Float, nullable=False, default=0.0),
    Column("status", String(20), nullable=False),  # ok | error
    Column("error", String(200)),
)

# Documents fetched by the scheduled collector (src/collector/); the content
# hash makes ingestion idempotent.
collected_documents = Table(
    "collected_documents",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("url", String(1000), nullable=False),
    Column("content_hash", String(64), nullable=False, unique=True),
    Column("doc_name", String(300), nullable=False),
    Column("chars", Integer, nullable=False),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
)

# Views for the agent's read-only SQL tool: only non-sensitive columns are exposed.
VIEWS_SQL = [
    """CREATE VIEW IF NOT EXISTS v_invoices AS
       SELECT id, company_id, company_name, invoice_number, issue_date, total_amount, currency,
              needs_human_review FROM invoices""",
    "CREATE VIEW IF NOT EXISTS v_companies AS SELECT id, name, voen FROM companies",
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@lru_cache(maxsize=4)
def _engine_for(url: str) -> Engine:
    if url.startswith("sqlite"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, future=True, pool_pre_ping=True)


def get_engine(url: str | None = None) -> Engine:
    return _engine_for(url or get_database_url())


def is_postgres(engine: Engine) -> bool:
    return engine.dialect.name == "postgresql"


def init_db(engine: Engine | None = None) -> Engine:
    """Create tables and views if missing (idempotent)."""
    engine = engine or get_engine()
    metadata.create_all(engine)
    with engine.begin() as conn:
        for statement in VIEWS_SQL:
            # SQLite and PostgreSQL both accept CREATE OR REPLACE-style via IF NOT EXISTS
            # only on SQLite; PostgreSQL needs CREATE OR REPLACE VIEW.
            conn.execute(text(statement if not is_postgres(engine) else statement.replace(
                "CREATE VIEW IF NOT EXISTS", "CREATE OR REPLACE VIEW")))
        if is_postgres(engine):
            # Grant the read-only role (created by docker/postgres/init.sh) access to the views only.
            has_role = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = 'assistant_ro'")).first()
            if has_role:
                conn.execute(text("GRANT SELECT ON v_invoices, v_companies TO assistant_ro"))
    return engine


def seed_companies(engine: Engine | None = None, path: Path | None = None) -> int:
    """Load the reference companies from JSON; existing ids are left untouched."""
    engine = engine or get_engine()
    path = path or PROJECT_ROOT / "data" / "reference" / "companies.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    with engine.begin() as conn:
        existing = {r[0] for r in conn.execute(select(companies.c.id))}
        new_rows = [r for r in rows if r["id"] not in existing]
        if new_rows:
            conn.execute(insert(companies), new_rows)
    return len(new_rows)


def save_invoice(data: dict, engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    with engine.begin() as conn:
        result = conn.execute(insert(invoices).values(created_at=utcnow(), **data))
        return int(result.inserted_primary_key[0])


def log_request_row(question: str, answer: str, sources: list[dict], pipeline: str, latency_ms: float,
                    engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(
            insert(request_logs).values(
                timestamp=utcnow(), question=question, answer=answer,
                sources_json=json.dumps(sources, ensure_ascii=False), pipeline=pipeline, latency_ms=latency_ms,
            )
        )


def count_rows(table: Table, engine: Engine | None = None) -> int:
    engine = engine or get_engine()
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())
