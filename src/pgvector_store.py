"""PostgreSQL + pgvector backend for document chunks (VECTOR_BACKEND=pgvector).

Exposes the same `similarity_search(query, k)` as the Chroma store, returning
LangChain Documents with `source`, `page` and `chunk_id` metadata, so the RAG
pipeline, the agent and the citation logic work unchanged.

Table:
    chunks(id, doc_name, page, chunk_id, content, embedding vector(N))
Index:
    HNSW on the embedding with cosine distance (vector_cosine_ops).
"""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config import EMBEDDING_DIM
from src.db import get_engine, is_postgres

logger = logging.getLogger(__name__)

BATCH_SIZE = 200


def to_vector_literal(vector: list[float]) -> str:
    """pgvector accepts '[0.1,0.2,...]' text; casting it with ::vector avoids needing a driver adapter."""
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"


def check_identifier(name: str) -> str:
    """Table names are put into SQL text (they cannot be bound parameters), so only plain names are allowed."""
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError(f"invalid table name {name!r}")
    return name


def create_schema_sql(table: str = "chunks", dim: int = EMBEDDING_DIM) -> list[str]:
    table = check_identifier(table)
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""CREATE TABLE IF NOT EXISTS {table} (
            id BIGSERIAL PRIMARY KEY,
            doc_name TEXT NOT NULL,
            page INTEGER,
            chunk_id TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL,
            embedding vector({dim}) NOT NULL
        )""",
        # HNSW = approximate nearest-neighbour graph: fast search, built once.
        f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw ON {table} USING hnsw (embedding vector_cosine_ops)",
    ]


SEARCH_SQL = """
    SELECT doc_name, page, chunk_id, content
    FROM {table}
    ORDER BY embedding <=> CAST(:query AS vector)
    LIMIT :k
"""


class PGVectorStore:
    def __init__(self, embeddings: Embeddings, engine: Engine | None = None, table: str = "chunks"):
        self.table = check_identifier(table)
        self.embeddings = embeddings
        self.engine = engine or get_engine()
        if not is_postgres(self.engine):
            raise RuntimeError(
                "VECTOR_BACKEND=pgvector needs PostgreSQL: set DATABASE_URL, e.g. "
                "postgresql://assistant:assistant@localhost:5432/assistant"
            )

    def create_schema(self) -> None:
        with self.engine.begin() as conn:
            for statement in create_schema_sql(self.table):
                conn.execute(text(statement))

    def reset(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {self.table}"))
        self.create_schema()

    def add_documents(self, chunks: list[Document]) -> int:
        self.create_schema()
        insert_sql = text(
            f"INSERT INTO {self.table} (doc_name, page, chunk_id, content, embedding) "
            "VALUES (:doc_name, :page, :chunk_id, :content, CAST(:embedding AS vector)) "
            "ON CONFLICT (chunk_id) DO NOTHING"
        )
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            vectors = self.embeddings.embed_documents([c.page_content for c in batch])
            rows = [
                {
                    "doc_name": c.metadata.get("source", "unknown"),
                    "page": c.metadata.get("page"),
                    "chunk_id": c.metadata.get("chunk_id") or f"{c.metadata.get('source')}::{start + i}",
                    "content": c.page_content,
                    "embedding": to_vector_literal(v),
                }
                for i, (c, v) in enumerate(zip(batch, vectors, strict=True))
            ]
            with self.engine.begin() as conn:
                conn.execute(insert_sql, rows)
        return len(chunks)

    def delete_source(self, doc_name: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {self.table} WHERE doc_name = :name"), {"name": doc_name})

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        with self.engine.connect() as conn:
            rows = conn.execute(text(SEARCH_SQL.format(table=self.table)), {"query": to_vector_literal(vector), "k": k}).all()
        return [
            Document(
                page_content=row.content,
                metadata={"source": row.doc_name, "page": row.page, "chunk_id": row.chunk_id},
            )
            for row in rows
        ]

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(text(f"SELECT count(*) FROM {self.table}")).scalar_one())
