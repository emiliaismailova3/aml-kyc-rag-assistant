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

from src.agent import calculator, internet_search, search_knowledge_base


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
