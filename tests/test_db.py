"""Tests for the database layer (src/db.py) and the pgvector backend.

SQLite-based tests always run. Tests marked `postgres` need the docker-compose
Postgres (`docker compose --profile full up -d postgres`) and are skipped when
it is not reachable.
"""

import json
import os
from datetime import date
from decimal import Decimal

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, text

from src import db
from src.config import get_database_url
from src.pgvector_store import PGVectorStore, check_identifier, create_schema_sql, to_vector_literal

TEST_PG_URL = os.getenv("TEST_DATABASE_URL", "postgresql+psycopg://assistant:assistant@localhost:5432/assistant")


@pytest.fixture
def sqlite_engine(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    db.init_db(engine)
    return engine


# --- config -------------------------------------------------------------------

def test_database_url_defaults_to_local_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert get_database_url().startswith("sqlite:///")


def test_database_url_gets_the_psycopg_driver(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host:5432/db")
    assert get_database_url() == "postgresql+psycopg://u:p@host:5432/db"


# --- relational tables (SQLite) ------------------------------------------------

def test_seed_companies_is_idempotent(sqlite_engine):
    assert db.seed_companies(sqlite_engine) == 50
    assert db.seed_companies(sqlite_engine) == 0
    assert db.count_rows(db.companies, sqlite_engine) == 50


def test_save_invoice_roundtrip(sqlite_engine):
    invoice_id = db.save_invoice(
        {"file_name": "a.pdf", "company_name": "X MMC", "voen": "1234567890", "invoice_number": "1",
         "issue_date": date(2026, 1, 2), "total_amount": Decimal("10.50"), "currency": "AZN",
         "needs_human_review": False},
        sqlite_engine,
    )
    assert invoice_id == 1
    assert db.count_rows(db.invoices, sqlite_engine) == 1


def test_views_hide_the_voen_of_invoices(sqlite_engine):
    with sqlite_engine.connect() as conn:
        columns = [row[1] for row in conn.execute(text("PRAGMA table_info(v_invoices)"))]
    assert "voen" not in columns and "total_amount" in columns


def test_request_log_row_is_stored(sqlite_engine):
    db.log_request_row("q", "a", [{"source": "s"}], "rag", 12.5, sqlite_engine)
    assert db.count_rows(db.request_logs, sqlite_engine) == 1


# --- pgvector helpers (no database needed) --------------------------------------

def test_vector_literal_format():
    assert to_vector_literal([0.5, -1.0]) == "[0.5000000,-1.0000000]"


def test_schema_sql_has_vector_column_and_hnsw_index():
    sql = " ".join(create_schema_sql("chunks", 384))
    assert "vector(384)" in sql and "USING hnsw (embedding vector_cosine_ops)" in sql


@pytest.mark.parametrize("name", ["chunks; DROP TABLE x", "Chunks", "1abc", "a-b", ""])
def test_table_name_must_be_a_plain_identifier(name):
    with pytest.raises(ValueError):
        check_identifier(name)


def test_pgvector_store_refuses_sqlite(sqlite_engine):
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        PGVectorStore(embeddings=None, engine=sqlite_engine)


# --- real PostgreSQL ---------------------------------------------------------------

class FakeEmbeddings(Embeddings):
    """Word-bag vectors in 384 dims: texts sharing words are close in cosine distance."""

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text_):
        vector = [0.0] * 384
        for word in text_.lower().split():
            vector[hash(word) % 384] += 1.0
        return vector


@pytest.fixture
def pg_store():
    try:
        engine = create_engine(TEST_PG_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL is not reachable (docker compose --profile full up -d postgres)")
    store = PGVectorStore(FakeEmbeddings(), engine=engine, table="chunks_test")
    store.reset()
    yield store
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS chunks_test"))


@pytest.mark.postgres
def test_pgvector_search_returns_the_closest_chunk_with_citation_metadata(pg_store):
    chunks = [
        Document(page_content="beneficial owner holds shares", metadata={"source": "a.pdf", "page": 3, "chunk_id": "a::1"}),
        Document(page_content="sanctions screening list", metadata={"source": "b.pdf", "page": 7, "chunk_id": "b::1"}),
    ]
    assert pg_store.add_documents(chunks) == 2
    hits = pg_store.similarity_search("who is the beneficial owner", k=1)
    assert hits[0].metadata == {"source": "a.pdf", "page": 3, "chunk_id": "a::1"}


@pytest.mark.postgres
def test_pgvector_ingestion_is_idempotent(pg_store):
    chunk = Document(page_content="same text", metadata={"source": "a.pdf", "page": 1, "chunk_id": "a::1"})
    pg_store.add_documents([chunk])
    pg_store.add_documents([chunk])
    assert pg_store.count() == 1


@pytest.mark.postgres
def test_postgres_schema_and_read_only_role():
    try:
        engine = create_engine(TEST_PG_URL)
        db.init_db(engine)
    except Exception:
        pytest.skip("PostgreSQL is not reachable")
    ro_url = TEST_PG_URL.replace("assistant:assistant@", "assistant_ro:assistant_ro@")
    ro = create_engine(ro_url)
    with ro.connect() as conn:
        conn.execute(text("SELECT count(*) FROM v_companies"))
        with pytest.raises(Exception):  # noqa: B017 - any permission error is the point
            conn.execute(text("SELECT count(*) FROM invoices"))


def test_seed_file_is_valid_json():
    rows = json.loads((db.PROJECT_ROOT / "data" / "reference" / "companies.json").read_text(encoding="utf-8"))
    assert {"id", "name", "voen"} <= set(rows[0])
