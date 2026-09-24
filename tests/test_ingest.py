"""Tests for the PDF ingestion pipeline (src/ingest.py)."""

from src.ingest import DEFAULT_RAW_DIR, load_pdfs, split_documents


def test_load_pdfs_returns_documents_with_source_metadata():
    documents = load_pdfs(DEFAULT_RAW_DIR)
    assert len(documents) > 0
    for doc in documents[:5]:
        assert doc.metadata["source"].endswith(".pdf")
        assert isinstance(doc.metadata["page"], int)
        assert doc.metadata["page"] >= 1


def test_split_documents_preserves_metadata_and_adds_chunk_id():
    documents = load_pdfs(DEFAULT_RAW_DIR)[:5]
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=50)

    assert len(chunks) >= len(documents)
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith(chunk.metadata["source"])
        assert len(chunk.page_content) <= 300 + 50  # allows for splitter overshoot on separators


def test_split_documents_respects_chunk_size_roughly():
    documents = load_pdfs(DEFAULT_RAW_DIR)[:3]
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=100)
    oversized = [c for c in chunks if len(c.page_content) > 700]
    assert len(oversized) == 0, "chunks should not wildly exceed the configured chunk_size"
