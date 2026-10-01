"""Tests for pure helper logic in src/rag.py (no LLM API key needed).

RAGPipeline.answer() itself requires a live LLM call and is exercised
manually (see README); these cover the two bugs found in live testing that
are easy to regress silently: showing sources alongside a refusal, and
leaving non-standard citation markup (observed from gpt-oss) in the answer.
"""

import pytest

from src.rag import clean_citations, is_refusal


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
