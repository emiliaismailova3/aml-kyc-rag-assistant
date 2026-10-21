"""Two-stage retrieval: a fast vector search for candidates, then a cross-encoder reranker.

The vector search compares the question and each chunk as two separate embeddings, which is
fast but coarse. A cross-encoder reads the question and the chunk *together* and scores how well
the chunk answers it, which is slower but more precise. So: fetch RERANK_CANDIDATES (12) chunks by
vector similarity, rescore them with the cross-encoder, keep the best k.

This matters most for the agent, which can only afford 4 chunks per search (Groq's free tier caps
a request at 8000 tokens): without reranking the right passage often ranked 5th-8th and the agent
kept searching until it hit its step limit.

Settings (.env): RERANK_ENABLED (default 1), RERANK_MODEL, RERANK_CANDIDATES (default 12).
If the model cannot be loaded, retrieval falls back to plain vector order and logs a warning.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from functools import lru_cache

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
Scorer = Callable[[str, Sequence[str]], Sequence[float]]


def rerank_enabled() -> bool:
    return os.getenv("RERANK_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def candidate_count() -> int:
    return int(os.getenv("RERANK_CANDIDATES", "12"))


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    name = os.getenv("RERANK_MODEL", DEFAULT_MODEL)
    logger.info("Loading reranker %s (first use downloads it)", name)
    # Our chunks are ~700 characters (~175 tokens); 256 tokens fits question + chunk and is
    # about twice as fast on CPU as the default 512.
    return CrossEncoder(name, device="cpu", max_length=256)


def cross_encoder_scores(query: str, passages: Sequence[str]) -> Sequence[float]:
    return _cross_encoder().predict([(query, p) for p in passages], show_progress_bar=False)


def rerank(query: str, docs: list[Document], k: int, scorer: Scorer | None = None) -> list[Document]:
    """Return the k docs the scorer rates highest (stable for ties). On scorer failure: first k as given."""
    if len(docs) <= 1:
        return docs[:k]
    try:
        scores = list((scorer or cross_encoder_scores)(query, [d.page_content for d in docs]))
    except Exception:  # noqa: BLE001 - a missing model must not take retrieval down
        logger.warning("Reranker unavailable; using vector-search order", exc_info=True)
        return docs[:k]
    order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
    return [docs[i] for i in order[:k]]


def retrieve(store, query: str, k: int, scorer: Scorer | None = None) -> list[Document]:
    """Vector search, then rerank when enabled. `store` is any object with similarity_search(query, k)."""
    if not rerank_enabled():
        return store.similarity_search(query, k=k)
    candidates = store.similarity_search(query, k=max(k, candidate_count()))
    return rerank(query, candidates, k, scorer)
