"""Tests for the agent's tools (src/agent.py).

The AgentExecutor itself (LLM-driven tool routing) requires a working
LLM_PROVIDER + API key, so it is exercised end-to-end via
data/agent_test_scenarios.json once a key is configured (see README). These
tests instead cover each tool function directly and in isolation: the
calculator's safe-eval behaviour (including that it rejects anything that
isn't arithmetic), and the knowledge-base search tool against the real,
already-built Chroma index from Step 4.
"""

import pytest
from langchain_core.messages import AIMessage

from src.agent import (
    _message_text,
    _sources_from_kb_output,
    calculator,
    internet_search,
    search_knowledge_base,
)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("450 * 37", "16650"),
        ("1 / 8", str(1 / 8)),
        ("12.5 - 12", str(12.5 - 12)),
        ("2 ** 10", "1024"),
        ("(2 + 3) * 4", "20"),
        ("-5 + 10", "5"),
    ],
)
def test_calculator_arithmetic(expression, expected):
    result = calculator.invoke({"expression": expression})
    assert result == expected


@pytest.mark.parametrize(
    "malicious_input",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "[].__class__.__base__.__subclasses__()",
        # Valid arithmetic that would hang the process (a ~370-million-digit result).
        "9 ** 9 ** 9",
        "True + True",
        "1 + " * 100 + "1",
    ],
)
def test_calculator_rejects_non_arithmetic(malicious_input):
    result = calculator.invoke({"expression": malicious_input})
    assert result.startswith("Error evaluating expression")


def test_search_knowledge_base_returns_relevant_chunk():
    result = search_knowledge_base.invoke({"query": "What does beneficial ownership mean?"})
    assert "wolfsberg_faqs_beneficial_ownership.pdf" in result
    assert "beneficial ownership" in result.lower() or "beneficial owner" in result.lower()


def test_search_knowledge_base_cites_source_and_page():
    result = search_knowledge_base.invoke({"query": "sanctions screening controls"})
    assert "Source:" in result
    assert "p." in result


def test_internet_search_returns_string_without_crashing():
    """Smoke test for the ddgs migration (duckduckgo_search was renamed
    upstream and had started returning irrelevant/localized junk for some
    queries). Network-dependent, so the assertion stays loose: it must not
    hit the tool's own exception-handling branch, but "no results for this
    exact query" is an acceptable, non-flaky outcome."""
    result = internet_search.invoke({"query": "FATF grey list jurisdictions"})
    assert isinstance(result, str)
    assert "Web search failed" not in result


KB_OUTPUT = """[1] (Source: fatf_recommendations_2025.pdf, p. 12)
Customer due diligence measures ...

[2] (Source: wolfsberg_faqs_pep.pdf, p. 3)
A politically exposed person ...

[3] (Source: fatf_recommendations_2025.pdf, p. 31)
Beneficial ownership ..."""


def test_agent_sources_map_citations_to_documents():
    sources = _sources_from_kb_output("PEPs need enhanced due diligence [2][3].", KB_OUTPUT)
    assert sources == [
        {"ref": 2, "source": "wolfsberg_faqs_pep.pdf", "page": 3},
        {"ref": 3, "source": "fatf_recommendations_2025.pdf", "page": 31},
    ]


def test_agent_sources_empty_for_refusal():
    assert _sources_from_kb_output("I don't know based on the available information.", KB_OUTPUT) == []


def test_message_text_handles_list_content_blocks():
    message = AIMessage(content=[{"type": "text", "text": "The product is "}, {"type": "text", "text": "16,650."}])
    assert _message_text(message) == "The product is 16,650."
