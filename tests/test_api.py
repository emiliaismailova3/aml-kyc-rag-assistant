"""Tests for the FastAPI /ask endpoint (src/api.py).

The retrieval + vector store stack is real (built in Step 4), but the LLM
call inside RAGPipeline.answer is monkeypatched so these tests exercise the
API's request/response contract and SQLite logging without requiring a paid
LLM API key or network access. End-to-end generation quality is covered
separately by src/evaluate.py (Step 8) once an LLM key is configured.
"""

from fastapi.testclient import TestClient

import src.api as api_module
from src.api import app

client = TestClient(app)


class _FakePipeline:
    def answer(self, question: str, k: int = 4):
        return {
            "question": question,
            "answer": "Beneficial ownership means ultimate control over funds. (Source: wolfsberg_faqs_beneficial_ownership.pdf, p. 1)",
            "sources": [{"ref": 1, "source": "wolfsberg_faqs_beneficial_ownership.pdf", "page": 1}],
        }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(api_module, "_pipeline", _FakePipeline())
    response = client.post("/ask", json={"question": "What is beneficial ownership?"})
    assert response.status_code == 200
    data = response.json()
    assert "Beneficial ownership" in data["answer"]
    assert data["sources"] == [{"ref": 1, "source": "wolfsberg_faqs_beneficial_ownership.pdf", "page": 1}]
    assert data["latency_ms"] >= 0


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_logs_to_sqlite(monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "_pipeline", _FakePipeline())

    logged = {}

    def fake_log_request(**kwargs):
        logged.update(kwargs)

    monkeypatch.setattr(api_module, "log_request", fake_log_request)

    client.post("/ask", json={"question": "What is beneficial ownership?"})
    assert logged["pipeline"] == "rag"
    assert logged["question"] == "What is beneficial ownership?"


def test_rate_limit_from_llm_provider_returns_clear_429(monkeypatch):
    import httpx
    from openai import RateLimitError

    class QuotaPipeline:
        def answer(self, question, k=4):
            response = httpx.Response(429, request=httpx.Request("POST", "https://api.example/chat"))
            raise RateLimitError("quota", response=response, body=None)

    monkeypatch.setattr(api_module, "_pipeline", QuotaPipeline())
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 429
    assert "quota" in response.json()["detail"].lower()


def test_ask_agent_returns_cited_sources_and_tool_calls(monkeypatch):
    class FakeAgent:
        def answer(self, question):
            return {
                "question": question,
                "answer": "PEPs need enhanced due diligence [2].",
                "tool_calls": [{"tool": "search_knowledge_base", "input": {"query": "PEP"}, "output": "..."}],
                "sources": [{"ref": 2, "source": "wolfsberg_faqs_pep.pdf", "page": 3}],
                "contexts": ["..."],
            }

    monkeypatch.setattr(api_module, "_agent_pipeline", FakeAgent())
    monkeypatch.setattr(api_module, "log_request", lambda **kwargs: None)
    response = client.post("/ask_agent", json={"question": "What about PEPs?"})
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == [{"ref": 2, "source": "wolfsberg_faqs_pep.pdf", "page": 3}]
    assert data["tool_calls"][0]["tool"] == "search_knowledge_base"
