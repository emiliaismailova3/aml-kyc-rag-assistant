"""Document ingestion pipeline for the AML/KYC knowledge base.

Loads every PDF in data/raw/, splits each page into overlapping chunks with
RecursiveCharacterTextSplitter, and attaches traceable metadata (source file
name, page number, and a stable chunk id) to every chunk so that answers can
always be cited back to "file, page N".

Usage:
    python -m src.ingest
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Chunk size / overlap chosen to keep each chunk within a coherent paragraph
# or two of regulatory text (long enough for context, short enough that
# retrieval doesn't drown the LLM in irrelevant surrounding text).
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def load_pdfs(raw_dir: Path = DEFAULT_RAW_DIR) -> list[Document]:
    """Load every PDF file in raw_dir into one Document per page.

    Each Document's metadata includes:
        source: the PDF's file name (e.g. "fatf_recommendations_2025.pdf")
        page: 1-indexed page number within that file
    """
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            # PyPDFLoader sets metadata["page"] 0-indexed; normalize to 1-indexed
            # and pin "source" to just the file name (not the full local path),
            # since the full path isn't portable across machines.
            page.metadata["source"] = pdf_path.name
            page.metadata["page"] = page.metadata.get("page", 0) + 1
        documents.extend(pages)
        logger.info("Loaded %d pages from %s", len(pages), pdf_path.name)

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split page-level documents into overlapping chunks with a stable id.

    Preserves the "source" and "page" metadata from the parent page and adds
    "chunk_id" (e.g. "fatf_recommendations_2025.pdf::p14::c2") so retrieved
    chunks can be traced back to an exact file, page, and chunk index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    chunk_index_per_source: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk_index_per_source[source] = chunk_index_per_source.get(source, 0) + 1
        page = chunk.metadata.get("page", "?")
        chunk.metadata["chunk_id"] = f"{source}::p{page}::c{chunk_index_per_source[source]}"

    return chunks


def ingest(
    raw_dir: Path = DEFAULT_RAW_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Load all PDFs in raw_dir and split them into retrieval-ready chunks."""
    documents = load_pdfs(raw_dir)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    n_sources = len({d.metadata["source"] for d in documents})
    logger.info(
        "Ingested %d source files -> %d pages -> %d chunks",
        n_sources,
        len(documents),
        len(chunks),
    )
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result_chunks = ingest()
    print(f"\nTotal chunks: {len(result_chunks)}")
    print("\nSample chunks:")
    for sample in result_chunks[:3]:
        print("---")
        print(sample.metadata)
        print(sample.page_content[:200].replace("\n", " "))
