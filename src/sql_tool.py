"""Safe, read-only SQL access for the agent.

Defence in depth (any one layer alone would not be enough):
  1. Parse with sqlglot and allow exactly one plain SELECT.
  2. Every table must be in a whitelist of two views (v_invoices, v_companies),
     which expose no sensitive columns (no VOEN on invoices).
  3. A LIMIT is mandatory: added if missing, rejected if above MAX_LIMIT.
  4. Dangerous functions (pg_sleep, set_config, ...) are rejected.
  5. Execution happens in a read-only transaction with a statement timeout, and
     on PostgreSQL through a role (assistant_ro) that can only read those views.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

import sqlglot
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlglot import exp
from sqlglot.errors import SqlglotError

from src.config import get_database_url
from src.db import is_postgres

logger = logging.getLogger(__name__)

ALLOWED_TABLES = {"v_invoices", "v_companies"}
DEFAULT_LIMIT = 50
MAX_LIMIT = 100
TIMEOUT_MS = 3000
MAX_OUTPUT_CHARS = 4000

BLOCKED_FUNCTIONS = {"set_config", "current_setting", "dblink", "lo_import", "lo_export", "load_extension", "readfile"}


class SQLNotAllowed(ValueError):
    """The query is not a safe read-only SELECT over the allowed views."""


def validate_sql(sql: str, dialect: str = "postgres") -> str:
    """Return the query to run (with a LIMIT guaranteed), or raise SQLNotAllowed."""
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except SqlglotError as exc:
        raise SQLNotAllowed(f"could not parse the SQL: {exc}") from exc
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SQLNotAllowed("send exactly one SQL statement")
    tree = statements[0]

    if not isinstance(tree, exp.Select):
        raise SQLNotAllowed("only a plain SELECT is allowed")
    if tree.find(exp.Into) or tree.args.get("locks"):
        raise SQLNotAllowed("SELECT ... INTO and row locking are not allowed")

    for table in tree.find_all(exp.Table):
        if table.name.lower() not in ALLOWED_TABLES or (table.db and table.db.lower() != "public"):
            raise SQLNotAllowed(
                f"table {table.sql()!r} is not available; allowed tables: {', '.join(sorted(ALLOWED_TABLES))}"
            )

    for func in tree.find_all(exp.Func):
        name = (func.name or "").lower()
        if name.startswith("pg_") or name in BLOCKED_FUNCTIONS:
            raise SQLNotAllowed(f"function {name!r} is not allowed")

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(DEFAULT_LIMIT)
    else:
        value = limit.expression
        if not (isinstance(value, exp.Literal) and value.is_int):
            raise SQLNotAllowed("LIMIT must be a plain integer")
        if int(value.name) > MAX_LIMIT:
            raise SQLNotAllowed(f"LIMIT must be at most {MAX_LIMIT}")
    return tree.sql(dialect=dialect)


@lru_cache(maxsize=1)
def _readonly_engine() -> Engine:
    """Prefer the dedicated read-only connection; fall back to the main one (SQLite, local dev)."""
    url = os.getenv("DATABASE_READONLY_URL", "").strip()
    if url:
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return create_engine(url, future=True, pool_pre_ping=True)
    # A separate engine (not the shared one): SQLite's read-only flag is per connection,
    # and must never leak into connections that write the request/LLM logs.
    return create_engine(get_database_url(), future=True)


def run_readonly_query(sql: str, engine: Engine | None = None) -> list[dict]:
    engine = engine or _readonly_engine()
    dialect = "postgres" if is_postgres(engine) else "sqlite"
    safe_sql = validate_sql(sql, dialect)
    with engine.connect() as conn:
        if is_postgres(engine):
            conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            conn.exec_driver_sql(f"SET LOCAL statement_timeout = {TIMEOUT_MS}")
            rows = conn.execute(text(safe_sql)).mappings().all()
        else:
            conn.exec_driver_sql("PRAGMA query_only = ON")
            try:
                rows = conn.execute(text(safe_sql)).mappings().all()
            finally:
                conn.exec_driver_sql("PRAGMA query_only = OFF")  # the connection goes back to a pool
    return [dict(row) for row in rows]


def format_rows(rows: list[dict]) -> str:
    if not rows:
        return "The query returned no rows."
    out = json.dumps(rows, default=str, ensure_ascii=False)
    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS] + f"... [truncated; {len(rows)} rows in total, use a smaller LIMIT]"
    return out
