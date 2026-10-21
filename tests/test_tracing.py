"""Tests for the optional Langfuse tracing (src/tracing.py). A fake client replaces Langfuse."""

import random
from contextlib import contextmanager

import pytest

from src import tracing
from src.config import ProviderSettings
from src.llm_client import LLMClient, LLMUnavailable, Provider, RawReply


class FakeObservation:
    def __init__(self, name, as_type, fields):
        self.name, self.as_type, self.fields, self.updates = name, as_type, fields, []

    def update(self, **fields):
        self.updates.append(fields)


class FakeLangfuse:
    def __init__(self):
        self.observations = []

    def start_as_current_observation(self, *, name, as_type="span", **fields):
        observation = FakeObservation(name, as_type, fields)
        self.observations.append(observation)

        @contextmanager
        def manager():
            yield observation

        return manager()


@pytest.fixture
def fake(monkeypatch):
    client = FakeLangfuse()
    monkeypatch.setattr(tracing, "get_langfuse", lambda: client)
    return client


class OkProvider(Provider):
    def __init__(self, script):
        super().__init__(ProviderSettings("p", "openai", "k", "", "big", "small"))
        self.script = list(script)

    def call(self, messages, model, max_tokens):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return RawReply(text=item, input_tokens=100, output_tokens=20)


def make_client(script):
    return LLMClient([OkProvider(script)], recorder=None, sleep=lambda s: None, rng=random.Random(1))


def test_tracing_is_off_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert not tracing.tracing_enabled()
    with tracing.observe("x") as observation:
        assert observation is None
    assert tracing.langchain_callbacks() == []


def test_tracing_turns_on_with_both_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)
    assert tracing.tracing_enabled()
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "0")
    assert not tracing.tracing_enabled()


def test_llm_call_is_recorded_as_a_generation_with_usage_and_cost(fake):
    reply = make_client(["hello"]).complete([{"role": "user", "content": "hi"}], task="demo")
    [generation] = fake.observations
    assert generation.name == "llm:demo" and generation.as_type == "generation"
    update = generation.updates[-1]
    assert update["output"] == "hello" and update["model"] == reply.model
    assert update["usage_details"] == {"input": 100, "output": 20}
    assert update["metadata"]["attempts"] == 1


def test_a_failed_llm_call_still_raises_and_is_traced(fake):
    class Status401(Exception):
        status_code = 401

    with pytest.raises(LLMUnavailable):
        make_client([Status401()]).complete([{"role": "user", "content": "hi"}])
    assert fake.observations[0].name == "llm:chat"


def test_a_broken_tracing_backend_never_breaks_the_request(monkeypatch):
    class Broken:
        def start_as_current_observation(self, **kwargs):
            raise RuntimeError("langfuse down")

    monkeypatch.setattr(tracing, "get_langfuse", lambda: Broken())
    assert make_client(["fine"]).complete([{"role": "user", "content": "hi"}]).text == "fine"


def test_trace_data_is_masked():
    data = {"messages": [{"content": "IBAN AZ21NABZ00000000137010001944, phone +994 50 123 45 67"}]}
    masked = tracing.mask_data(data=data)
    text = masked["messages"][0]["content"]
    assert "[IBAN]" in text and "[PHONE]" in text and "AZ21NABZ" not in text
