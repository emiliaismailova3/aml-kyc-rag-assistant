"""Retrieval quality smoke test for the Chroma vector store (Step 4).

For a handful of representative AML/KYC questions, checks that a chunk from
the document we know contains the answer actually shows up in the top-3
results. This is not a full RAGAS evaluation (that's Step 8) -- it is a fast
sanity check that ingestion + embeddings + Chroma are wired together
correctly before building the RAG pipeline on top of them.

Requires the vector store to already be built:
    python -m src.vectorstore
"""

import pytest

from src.vectorstore import load_vectorstore

# (query, substring expected in at least one of the top-3 chunks' source file name)
RETRIEVAL_CASES = [
    (
        "What does beneficial ownership mean for AML purposes?",
        "wolfsberg_faqs_beneficial_ownership.pdf",
    ),
    (
        "What customer due diligence measures must financial institutions apply to every customer?",
        "fatf_recommendations_2025.pdf",
    ),
    (
        "How does the Wolfsberg Group define a politically exposed person (PEP)?",
        "wolfsberg_pep_guidance.pdf",
    ),
    (
        "What is the difference between source of wealth and source of funds?",
        "wolfsberg_source_of_wealth_funds_faqs.pdf",
    ),
    (
        "What are the two main sanctions screening controls used by financial institutions?",
        "wolfsberg_sanctions_screening_guidance.pdf",
    ),
    (
        "How long can Azerbaijan's financial monitoring organ freeze a suspicious transaction?",
        "azerbaijan_aml_law_unodc.pdf",
    ),
    (
        "How can I check if a payment institution in Azerbaijan is licensed by the Central Bank?",
        "cbar_notice_unlicensed_payment_institutions.pdf",
    ),
]


@pytest.fixture(scope="module")
def vectorstore():
    return load_vectorstore()


@pytest.mark.parametrize("query,expected_source", RETRIEVAL_CASES)
def test_relevant_chunk_in_top_3(vectorstore, query, expected_source):
    results = vectorstore.similarity_search(query, k=3)
    assert len(results) > 0, f"No results returned for query: {query!r}"

    sources = [doc.metadata.get("source", "") for doc in results]
    assert expected_source in sources, (
        f"Expected {expected_source!r} in top-3 sources for query {query!r}, "
        f"got {sources!r}"
    )
