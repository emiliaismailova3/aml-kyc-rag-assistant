"""One URL -> clean text -> dedupe -> chunks -> vector store (idempotent).

Running the same URL twice never duplicates anything:
  - same content            -> "duplicate" (its hash is already in collected_documents)
  - same URL, changed text  -> the old chunks of that document are replaced
  - new URL                 -> "ingested"
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from urllib.parse import urlparse

from langchain_core.documents import Document
from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.collector.clean import content_hash, extract_text
from src.collector.fetch import fetch
from src.db import collected_documents, get_engine, init_db, utcnow
from src.ingest import split_documents

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 200  # shorter than this is an error page / empty shell, not a document

Fetcher = Callable[[str], tuple[bytes, str]]


def doc_name_for(url: str, content_type: str) -> str:
    """A readable, stable file-like name used as the `source` shown in citations."""
    parts = urlparse(url)
    slug = re.sub(r"[^a-z0-9]+", "_", f"{parts.netloc}{parts.path}".lower()).strip("_")[:100]
    return f"{slug}.{'pdf' if content_type == 'application/pdf' else 'html'}"


def remove_document(store, doc_name: str) -> None:
    """Delete every chunk of one document from the vector store (Chroma or pgvector)."""
    if hasattr(store, "delete_source"):  # PGVectorStore
        store.delete_source(doc_name)
    else:  # Chroma
        store._collection.delete(where={"source": doc_name})


def add_chunks(store, chunks: list[Document]) -> None:
    if hasattr(store, "delete_source"):
        store.add_documents(chunks)  # pgvector: ON CONFLICT DO NOTHING
    else:
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])  # Chroma: upsert by id


def collect_url(
    url: str,
    fetcher: Fetcher = fetch,
    store=None,
    engine: Engine | None = None,
) -> dict:
    engine = init_db(engine or get_engine())
    body, content_type = fetcher(url)
    text = extract_text(body, content_type)
    if len(text) < MIN_TEXT_CHARS:
        logger.warning("%s produced only %d characters of text; skipping", url, len(text))
        return {"url": url, "status": "empty"}

    digest = content_hash(text)
    with engine.connect() as conn:
        if conn.execute(select(collected_documents.c.id).where(collected_documents.c.content_hash == digest)).first():
            return {"url": url, "status": "duplicate"}
        previous = conn.execute(
            select(collected_documents.c.id).where(collected_documents.c.url == url).limit(1)
        ).first()

    if store is None:
        from src.vectorstore import load_vectorstore

        store = load_vectorstore()

    name = doc_name_for(url, content_type)
    if previous:
        remove_document(store, name)  # the page changed: replace its old chunks
    chunks = split_documents([Document(page_content=text, metadata={"source": name, "page": 1})])
    add_chunks(store, chunks)
    with engine.begin() as conn:
        conn.execute(
            insert(collected_documents).values(
                url=url, content_hash=digest, doc_name=name, chars=len(text), fetched_at=utcnow()
            )
        )
    return {"url": url, "status": "updated" if previous else "ingested", "doc_name": name, "chunks": len(chunks)}
