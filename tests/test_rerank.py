"""Tests for two-stage retrieval (src/rerank.py). A fake scorer replaces the cross-encoder."""

from langchain_core.documents import Document

from src import rerank


class FakeStore:
    def __init__(self, docs):
        self.docs = docs
        self.asked_k = None

    def similarity_search(self, query, k=4):
        self.asked_k = k
        return self.docs[:k]


DOCS = [Document(page_content=f"chunk {i}", metadata={"i": i}) for i in range(20)]


def by_number_desc(query, passages):
    return [int(p.split()[1]) for p in passages]  # "chunk 7" scores 7: later chunks rank higher


def test_rerank_keeps_the_k_best_scored():
    top = rerank.rerank("q", DOCS[:12], k=4, scorer=by_number_desc)
    assert [d.metadata["i"] for d in top] == [11, 10, 9, 8]


def test_retrieve_fetches_candidates_then_reranks(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "1")
    monkeypatch.setenv("RERANK_CANDIDATES", "12")
    store = FakeStore(DOCS)
    top = rerank.retrieve(store, "q", k=4, scorer=by_number_desc)
    assert store.asked_k == 12
    assert [d.metadata["i"] for d in top] == [11, 10, 9, 8]


def test_retrieve_with_reranking_off_is_plain_vector_search(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "0")
    store = FakeStore(DOCS)
    top = rerank.retrieve(store, "q", k=4, scorer=by_number_desc)
    assert store.asked_k == 4 and [d.metadata["i"] for d in top] == [0, 1, 2, 3]


def test_a_broken_reranker_falls_back_to_vector_order():
    def broken(query, passages):
        raise OSError("model files missing")

    top = rerank.rerank("q", DOCS[:12], k=3, scorer=broken)
    assert [d.metadata["i"] for d in top] == [0, 1, 2]


def test_never_asks_for_fewer_candidates_than_k(monkeypatch):
    monkeypatch.setenv("RERANK_CANDIDATES", "2")
    store = FakeStore(DOCS)
    rerank.retrieve(store, "q", k=8, scorer=by_number_desc)
    assert store.asked_k == 8
