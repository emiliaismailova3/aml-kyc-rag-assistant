"""Embeddings and Chroma vector store for the AML/KYC knowledge base.

Supports two embedding backends, selected via EMBEDDING_PROVIDER in .env:
  - "local": sentence-transformers (BAAI/bge-small-en-v1.5 by default). Free,
    runs on CPU, needs no API key -- this is the default so the project is
    runnable out of the box.
  - "openai": OpenAI's text-embedding-3-small. Higher quality, needs
    OPENAI_API_KEY.

Usage:
    python -m src.vectorstore   # (re)builds the persisted Chroma index
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, VECTOR_BACKEND, get_embedding_config
from src.ingest import ingest

logger = logging.getLogger(__name__)


def get_embeddings() -> Embeddings:
    """Return the embedding model configured via EMBEDDING_PROVIDER in .env.

    Cached per configuration: loading the sentence-transformers model takes
    seconds, and the agent's knowledge-base tool used to reload it on every
    single tool call.
    """
    return _build_embeddings(get_embedding_config())


@lru_cache(maxsize=4)
def _build_embeddings(config) -> Embeddings:
    if config.provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Using local embedding model: %s", config.local_model)
        return HuggingFaceEmbeddings(
            model_name=config.local_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    if config.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not config.openai_api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set in .env"
            )
        logger.info("Using OpenAI embedding model: %s", config.openai_model)
        return OpenAIEmbeddings(
            model=config.openai_model,
            api_key=config.openai_api_key,
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={config.provider!r}; expected 'local' or 'openai'"
    )


def build_vectorstore(chunks: list[Document] | None = None, reset: bool = True):
    """Embed chunks and persist them to the configured vector store
    (Chroma by default, PostgreSQL + pgvector when VECTOR_BACKEND=pgvector).

    If chunks is None, runs the full ingestion pipeline (load PDFs + split)
    first. If reset is True (default), any existing collection with the same
    name is deleted first so re-running this script doesn't accumulate
    duplicate chunks.
    """
    if chunks is None:
        chunks = ingest()

    embeddings = get_embeddings()

    if VECTOR_BACKEND == "pgvector":
        from src.pgvector_store import PGVectorStore

        store = PGVectorStore(embeddings)
        if reset:
            store.reset()
        count = store.add_documents(chunks)
        logger.info("Persisted %d chunks to PostgreSQL (pgvector)", count)
        return store

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    if reset:
        # Deleting and recreating the collection keeps re-indexing idempotent:
        # running this script twice never duplicates chunks.
        try:
            existing = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_PERSIST_DIR),
            )
            existing.delete_collection()
            logger.info("Deleted existing collection %r before re-indexing", CHROMA_COLLECTION_NAME)
        except Exception:
            logger.debug("No existing collection to delete (first run)")

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )
    # Any cached handle now points at a collection that was just replaced.
    _load_vectorstore_cached.cache_clear()
    logger.info(
        "Persisted %d chunks to Chroma collection %r at %s",
        len(chunks),
        CHROMA_COLLECTION_NAME,
        CHROMA_PERSIST_DIR,
    )
    return store


def load_vectorstore():
    """Load the already-persisted vector store (does not re-index).

    The handle is cached and shared by the RAG pipeline and the agent's
    knowledge-base tool, so the embedding model and the Chroma client are
    opened once per process rather than once per question / tool call.
    """
    if VECTOR_BACKEND == "pgvector":
        return _load_pgvector_cached(get_embedding_config())
    return _load_vectorstore_cached(get_embedding_config())


@lru_cache(maxsize=4)
def _load_pgvector_cached(config):
    from src.pgvector_store import PGVectorStore

    return PGVectorStore(_build_embeddings(config))


@lru_cache(maxsize=4)
def _load_vectorstore_cached(config) -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=_build_embeddings(config),
        persist_directory=str(CHROMA_PERSIST_DIR),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_vectorstore()
