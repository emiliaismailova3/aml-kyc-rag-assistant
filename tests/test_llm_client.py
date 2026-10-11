"""Tests for the reliability layer: routing, retry with backoff, fallback chain, logging, /stats.

Providers are scripted fakes, and `sleep` is replaced so no test actually waits.
"""

import random

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

import src.api as api_module
from src import llm_stats
from src.config import ProviderSettings, get_prices, get_provider_chain
from src.db import init_db
from src.invoices.extract import extract_from_text
from src.llm_client import (
    LLMClient,
    LLMUnavailable,
    Provider,
    RawReply,
    backoff_delay,
    choose_tier,
    estimate_cost,
    is_retryable,
)
from src.rag import RAGPipeline


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeProvider(Provider):
    """Plays back a script: each item is a reply text, or an Exception to raise."""

    def __init__(self, name, script, main="big", small="small"):
        super().__init__(ProviderSettings(name, "openai", "key", "", main, small))
        self.script = list(script)
        self.calls = []

    def call(self, messages, model, max_tokens):
        self.calls.append(model)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return RawReply(text=item, input_tokens=1000, output_tokens=500)


def make_client(*providers, recorder=None):
    rows = []
    client = LLMClient(
        list(providers),
        recorder=recorder or rows.append,
        sleep=lambda seconds: rows.append({"slept": seconds}),
        rng=random.Random(1),
    )
    return client, rows


MESSAGES = [{"role": "user", "content": "hi"}]


# --- routing ------------------------------------------------------------------

def test_short_plain_prompt_goes_to_the_small_model():
    assert choose_tier(MESSAGES) == "small"


@pytest.mark.parametrize(
    "kwargs,messages",
    [
        ({"needs_tools": True}, MESSAGES),
        ({"retry": True}, MESSAGES),
        ({}, [{"role": "user", "content": "x" * 5000}]),
    ],
)
def test_tools_retries_and_long_prompts_go_to_the_main_model(kwargs, messages):
    assert choose_tier(messages, **kwargs) == "main"


def test_client_uses_the_model_that_matches_the_tier():
    provider = FakeProvider("p", ["a", "b"])
    client, _ = make_client(provider)
    client.complete(MESSAGES)
    client.complete(MESSAGES, retry=True)
    assert provider.calls == ["small", "big"]


# --- retry classification and backoff -------------------------------------------

@pytest.mark.parametrize("status,expected", [(429, True), (500, True), (503, True), (408, True),
                                              (400, False), (401, False), (404, False)])
def test_is_retryable_by_status(status, expected):
    assert is_retryable(StatusError(status)) is expected


def test_is_retryable_for_connection_problems_only_by_name():
    class APIConnectionError(Exception):
        pass

    assert is_retryable(APIConnectionError()) is True
    assert is_retryable(ValueError("bad")) is False


def test_backoff_grows_exponentially_and_is_capped_with_jitter():
    rng = random.Random(0)
    delays = [backoff_delay(n, rng) for n in range(8)]
    for n, delay in enumerate(delays):
        ceiling = min(30.0, 2**n)
        assert ceiling / 2 <= delay <= ceiling
    assert delays[7] <= 30.0


def test_transient_errors_are_retried_then_succeed():
    provider = FakeProvider("p", [StatusError(429), StatusError(503), "ok"])
    client, log = make_client(provider)
    reply = client.complete(MESSAGES)
    assert reply.text == "ok" and reply.attempts == 3 and not reply.fell_back
    assert len([r for r in log if "slept" in r]) == 2  # waited twice
    assert [r["status"] for r in log if "status" in r] == ["error", "error", "ok"]


def test_non_retryable_error_is_not_retried_on_the_same_provider():
    provider = FakeProvider("p", [StatusError(401), "never used"])
    client, _ = make_client(provider)
    with pytest.raises(LLMUnavailable):
        client.complete(MESSAGES)
    assert len(provider.calls) == 1


# --- fallback chain and escalation ------------------------------------------------

def test_falls_back_to_the_next_provider_after_retries_are_exhausted():
    first = FakeProvider("openai", [StatusError(503)] * 4)
    second = FakeProvider("anthropic", ["from backup"])
    client, _ = make_client(first, second)
    reply = client.complete(MESSAGES)
    assert reply.text == "from backup" and reply.provider == "anthropic" and reply.fell_back
    assert len(first.calls) == 4  # 1 try + 3 retries


def test_a_provider_with_a_bad_key_does_not_block_the_chain():
    client, _ = make_client(FakeProvider("a", [StatusError(401)]), FakeProvider("b", ["ok"]))
    assert client.complete(MESSAGES).provider == "b"


def test_all_providers_failing_raises_with_the_human_escalation_flag():
    client, _ = make_client(FakeProvider("a", [StatusError(401)]), FakeProvider("b", [StatusError(401)]))
    with pytest.raises(LLMUnavailable) as info:
        client.complete(MESSAGES)
    assert info.value.needs_human is True and not info.value.rate_limited


def test_exhausted_rate_limit_is_reported_as_rate_limited():
    client, _ = make_client(FakeProvider("a", [StatusError(429)] * 4))
    with pytest.raises(LLMUnavailable) as info:
        client.complete(MESSAGES)
    assert info.value.rate_limited is True


def test_empty_chain_means_no_key_configured():
    client, _ = make_client()
    with pytest.raises(LLMUnavailable, match="API key"):
        client.complete(MESSAGES)


# --- logging and cost ---------------------------------------------------------------

def test_every_attempt_is_recorded_without_prompt_text():
    client, log = make_client(FakeProvider("p", [StatusError(500), "secret answer"]))
    client.complete([{"role": "user", "content": "my secret question"}], task="demo")
    records = [r for r in log if "status" in r]
    assert [r["status"] for r in records] == ["error", "ok"]
    assert records[1]["task"] == "demo" and records[1]["input_tokens"] == 1000
    assert "secret" not in str(records)


def test_a_failing_recorder_never_breaks_the_request():
    def broken(row):
        raise RuntimeError("db down")

    client, _ = make_client(FakeProvider("p", ["fine"]), recorder=broken)
    assert client.complete(MESSAGES).text == "fine"


def test_cost_uses_the_price_table_and_unknown_models_cost_zero():
    assert estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.15 + 0.60)
    assert estimate_cost("some-unknown-model", 1000, 1000) == 0.0


def test_prices_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_PRICES_JSON", '{"my-model": [2.0, 4.0]}')
    assert get_prices()["my-model"] == (2.0, 4.0)


# --- configuration ---------------------------------------------------------------------

def test_provider_chain_skips_providers_without_a_key(monkeypatch):
    for name in ("GROQ", "OPENAI", "ANTHROPIC", "TOGETHER"):
        monkeypatch.delenv(f"{name}_API_KEY", raising=False)
    monkeypatch.setenv("LLM_FALLBACK_CHAIN", "openai,anthropic,groq")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    assert [p.name for p in get_provider_chain()] == ["anthropic", "groq"]


def test_provider_chain_defaults_to_the_primary_provider(monkeypatch):
    monkeypatch.delenv("LLM_FALLBACK_CHAIN", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("LLM_MODEL", "my-main-model")
    [provider] = get_provider_chain()
    assert provider.main_model == "my-main-model"


def test_unknown_provider_in_chain_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_CHAIN", "nope")
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider_chain()


# --- integration with the rest of the app ------------------------------------------------

def test_rag_pipeline_goes_through_the_client():
    class FakeClient:
        def complete(self, messages, task):
            self.task = task
            return type("R", (), {"text": "Owners are identified [1]."})()

    client = FakeClient()
    pipeline = RAGPipeline(top_k=2, client=client)
    result = pipeline.answer("What is beneficial ownership?")
    assert client.task == "rag_answer" and result["answer"] == "Owners are identified [1]."
    assert result["sources"][0]["ref"] == 1


def test_invoice_extraction_flags_human_review_when_the_llm_is_down():
    def down(messages):
        raise LLMUnavailable("all providers failed")

    result = extract_from_text("invoice text", down)
    assert result.needs_human_review and result.invoice is None
    assert "all providers failed" in result.errors[0]


def test_api_returns_503_with_escalation_flag_when_llm_is_unavailable(monkeypatch):
    class DownPipeline:
        def answer(self, question, k=8):
            raise LLMUnavailable("all providers failed")

    monkeypatch.setattr(api_module, "_pipeline", DownPipeline())
    response = TestClient(api_module.app).post("/ask", json={"question": "hello?"})
    assert response.status_code == 503
    assert response.json()["escalate_to_human"] is True


def test_stats_endpoint_summarises_recorded_calls(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'stats.db').as_posix()}")
    init_db(engine)
    from sqlalchemy import insert

    from src.db import llm_calls, utcnow

    rows = [
        {"provider": "p", "model": "m", "latency_ms": 100.0 * i, "input_tokens": 10, "output_tokens": 5,
         "cost_usd": 0.01, "status": "ok"}
        for i in range(1, 11)
    ] + [{"provider": "p", "model": "m", "latency_ms": 5.0, "input_tokens": 0, "output_tokens": 0,
          "cost_usd": 0.0, "status": "error", "error": "RateLimitError"}]
    with engine.begin() as conn:
        conn.execute(insert(llm_calls), [{"timestamp": utcnow(), **r} for r in rows])

    monkeypatch.setattr(llm_stats, "get_engine", lambda: engine)
    body = TestClient(api_module.app).get("/stats").json()
    assert body["calls"] == 11
    assert body["error_rate"] == round(1 / 11, 4)
    assert body["latency_ms"]["p50"] == 550.0
    assert body["latency_ms"]["p95"] == 955.0
    assert body["total_cost_usd"] == pytest.approx(0.10)
    assert body["by_model"]["m"]["errors"] == 1


def test_stats_with_no_calls(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    assert llm_stats.get_stats(engine)["calls"] == 0
