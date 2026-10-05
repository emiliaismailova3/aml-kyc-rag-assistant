"""Tests for pure helper logic in src/rag.py (no LLM API key needed).

RAGPipeline.answer() itself requires a live LLM call and is exercised
manually (see README); these cover the two bugs found in live testing that
are easy to regress silently: showing sources alongside a refusal, and
leaving non-standard citation markup (observed from gpt-oss) in the answer.
"""

import pytest
from langchain_core.documents import Document

from src.rag import build_sources, cited_refs, clean_citations, is_refusal


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know based on the available documents.",
        "I don't know based on the available information.",
        "I DON'T KNOW based on the available documents.",
        "  I don't know based on the available documents.  ",
    ],
)
def test_is_refusal_true_for_refusal_variants(answer):
    assert is_refusal(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "Beneficial ownership means ultimate control over funds [1].",
        "I don't think that's covered, but here's what is relevant...",
        "",
    ],
)
def test_is_refusal_false_for_real_answers(answer):
    assert is_refusal(answer) is False


def test_clean_citations_normalizes_weird_brackets():
    raw = "Beneficial owners must be identified【3†Source: file.pdf, p. 12】."
    cleaned = clean_citations(raw)
    assert cleaned == "Beneficial owners must be identified[3]."


def test_clean_citations_leaves_plain_citations_alone():
    raw = "Beneficial owners must be identified [1] and verified [2]."
    assert clean_citations(raw) == raw


# --- Citation-aware sources -------------------------------------------------

def _chunks(n):
    return [Document(page_content=f"text {i}", metadata={"source": f"doc{i}.pdf", "page": i}) for i in range(1, n + 1)]


def test_cited_refs_keeps_order_dedupes_and_drops_out_of_range():
    answer = "Owners must be identified [5][3], verified [3, 1] and monitored [9]."
    assert cited_refs(answer, n_chunks=8) == [5, 3, 1]


def test_build_sources_returns_only_cited_chunks_with_their_numbers():
    sources = build_sources("Control means ultimate ownership [2][4].", _chunks(8))
    assert sources == [
        {"ref": 2, "source": "doc2.pdf", "page": 2},
        {"ref": 4, "source": "doc4.pdf", "page": 4},
    ]


def test_build_sources_falls_back_to_all_chunks_when_nothing_is_cited():
    sources = build_sources("An answer without citation markers.", _chunks(3))
    assert [s["ref"] for s in sources] == [1, 2, 3]


def test_build_sources_is_empty_for_a_refusal():
    assert build_sources("I don't know based on the available documents.", _chunks(3)) == []
