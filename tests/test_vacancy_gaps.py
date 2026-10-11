"""Tests for translation, summarization and streaming. LLM providers are scripted fakes: no keys, no network."""

import json
import random

import pytest
from fastapi.testclient import TestClient

import src.api as api_module
from src import text_tools
from src.config import ProviderSettings
from src.llm_client import LLMClient, LLMUnavailable, Provider, RawReply
from src.rag import RAGPipeline


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class StreamProvider(Provider):
    """Each script item is a list of pieces to stream; an Exception item is raised when the stream starts;
    an Exception inside the list is raised in the middle of the stream."""

    def __init__(self, name, script):
        super().__init__(ProviderSettings(name, "openai", "key", "", "big", "small"))
        self.script = list(script)

    def call(self, messages, model, max_tokens):
        return RawReply(text="".join(self.script.pop(0)), input_tokens=10, output_tokens=5)

    def stream(self, messages, model, max_tokens):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        for piece in item:
            if isinstance(piece, Exception):
                raise piece
            yield piece


def make_client(*providers):
    rows = []
    client = LLMClient(list(providers), recorder=rows.append, sleep=lambda s: None, rng=random.Random(1))
    return client, rows


MESSAGES = [{"role": "user", "content": "hello"}]


# --- LLMClient.stream -----------------------------------------------------------------

def test_stream_yields_pieces_and_logs_one_ok_call_with_estimated_tokens():
    client, rows = make_client(StreamProvider("p", [["Hel", "lo ", "world"]]))
    assert list(client.stream(MESSAGES, task="demo")) == ["Hel", "lo ", "world"]
    assert len(rows) == 1 and rows[0]["status"] == "ok" and rows[0]["task"] == "demo"
    assert rows[0]["output_tokens"] == len("Hello world") // 4


def test_stream_retries_before_the_first_piece():
    client, rows = make_client(StreamProvider("p", [StatusError(429), ["ok"]]))
    assert list(client.stream(MESSAGES)) == ["ok"]
    assert [r["status"] for r in rows] == ["error", "ok"]


def test_stream_falls_back_to_the_next_provider_before_the_first_piece():
    client, _ = make_client(StreamProvider("a", [StatusError(503)] * 4), StreamProvider("b", [["backup"]]))
    assert list(client.stream(MESSAGES)) == ["backup"]


def test_stream_that_breaks_midway_is_not_restarted_on_another_provider():
    second = StreamProvider("b", [["never used"]])
    client, _ = make_client(StreamProvider("a", [["Hel", StatusError(500)]]), second)
    received = []
    with pytest.raises(LLMUnavailable, match="broke"):
        for piece in client.stream(MESSAGES):
            received.append(piece)
    assert received == ["Hel"] and second.script == [["never used"]]


def test_stream_with_every_provider_failing_raises_unavailable():
    client, _ = make_client(StreamProvider("a", [StatusError(401)]))
    with pytest.raises(LLMUnavailable):
        list(client.stream(MESSAGES))


# --- translate / summarize -------------------------------------------------------------

class EchoClient:
    """Returns '<task>:<first 20 chars of the user text>' and remembers what it was asked."""

    def __init__(self):
        self.calls = []

    def complete(self, messages, task):
        self.calls.append((task, messages))
        return type("R", (), {"text": f" {task}:{messages[-1]['content'][:20]} "})()


def test_translate_names_the_target_language_in_the_system_prompt():
    client = EchoClient()
    result = text_tools.translate("Salam dünya", "ru", client=client)
    assert result == "translate:Salam dünya"
    assert "Russian" in client.calls[0][1][0]["content"]


def test_summarize_passes_language_and_sentence_limit():
    client = EchoClient()
    text_tools.summarize("A short report.", language="az", max_sentences=3, client=client)
    system = client.calls[0][1][0]["content"]
    assert "Azerbaijani" in system and "at most 3 sentences" in system


def test_long_text_is_summarized_map_reduce():
    client = EchoClient()
    text = "\n".join(f"Paragraph {i}. " + "x" * 900 for i in range(20))  # ~18,000 characters
    text_tools.summarize(text, client=client)
    pieces = text_tools.split_text(text)
    assert len(pieces) > 1
    assert len(client.calls) == len(pieces) + 1  # one call per piece + one to combine


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_text_is_rejected(bad):
    with pytest.raises(text_tools.TextToolError):
        text_tools.translate(bad, "en", client=EchoClient())


def test_unknown_language_and_oversized_text_are_rejected():
    with pytest.raises(text_tools.TextToolError, match="Unsupported language"):
        text_tools.translate("hi", "klingon", client=EchoClient())
    with pytest.raises(text_tools.TextToolError, match="longer than"):
        text_tools.summarize("x" * (text_tools.MAX_INPUT_CHARS + 1), client=EchoClient())


def test_split_text_respects_the_limit_even_for_one_huge_paragraph():
    pieces = text_tools.split_text("Sentence number one. " * 500, limit=1000)
    assert all(len(p) <= 1000 for p in pieces)
    assert "".join(p.replace(" ", "") for p in pieces) == ("Sentence number one. " * 500).replace(" ", "")


# --- endpoints ---------------------------------------------------------------------------

client = TestClient(api_module.app)


def test_translate_endpoint(monkeypatch):
    monkeypatch.setattr(api_module, "translate", lambda text, target: f"[{target}] {text}")
    assert client.post("/translate", json={"text": "hi", "target_language": "az"}).json() == {"translation": "[az] hi"}


def test_translate_endpoint_returns_422_for_a_bad_language():
    response = client.post("/translate", json={"text": "hi", "target_language": "klingon"})
    assert response.status_code == 422 and "Unsupported" in response.json()["detail"]


def test_summarize_endpoint(monkeypatch):
    monkeypatch.setattr(api_module, "summarize", lambda text, language, n: f"{language}/{n}/{text}")
    body = client.post("/summarize", json={"text": "doc", "language": "ru", "max_sentences": 2}).json()
    assert body == {"summary": "ru/2/doc"}


class FakeStreamingPipeline:
    def stream_answer(self, question, k=8):
        yield {"type": "token", "text": "Owners "}
        yield {"type": "token", "text": "are identified [1]."}
        yield {"type": "done", "answer": "Owners are identified [1].", "sources": [], "time_to_first_token_ms": 12.0}


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_ask_stream_sends_tokens_then_done(monkeypatch):
    monkeypatch.setattr(api_module, "_pipeline", FakeStreamingPipeline())
    response = client.post("/ask_stream", json={"question": "What is beneficial ownership?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["token", "token", "done"]
    assert events[-1][1]["answer"] == "Owners are identified [1]."


def test_ask_stream_returns_503_when_no_provider_works_before_the_first_token(monkeypatch):
    class Down:
        def stream_answer(self, question, k=8):
            raise LLMUnavailable("all providers failed")
            yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(api_module, "_pipeline", Down())
    response = client.post("/ask_stream", json={"question": "hello?"})
    assert response.status_code == 503 and response.json()["escalate_to_human"] is True


def test_ask_stream_reports_a_midstream_failure_as_an_error_event(monkeypatch):
    class Breaks:
        def stream_answer(self, question, k=8):
            yield {"type": "token", "text": "Owners "}
            raise LLMUnavailable("stream broke")

    monkeypatch.setattr(api_module, "_pipeline", Breaks())
    events = parse_sse(client.post("/ask_stream", json={"question": "hello?"}).text)
    assert [name for name, _ in events] == ["token", "error"] and events[-1][1]["escalate_to_human"] is True


def test_rag_stream_answer_emits_tokens_then_cleaned_done_event():
    class FakeLLM:
        def stream(self, messages, task):
            self.task = task
            yield from ["Owners are ", "identified 【1†Source: x】."]

    pipeline = RAGPipeline.__new__(RAGPipeline)  # skip loading the real vector store
    pipeline.top_k, pipeline._client = 2, FakeLLM()
    pipeline.retrieve = lambda question, k=None: []
    events = list(pipeline.stream_answer("q"))
    assert [e["type"] for e in events] == ["token", "token", "done"]
    assert events[-1]["answer"] == "Owners are identified [1]."
    assert events[-1]["time_to_first_token_ms"] is not None


# --- voice check (WER) -----------------------------------------------------------------

def test_word_error_rate():
    from scripts.check_voice import normalize, word_error_rate

    assert word_error_rate("What is beneficial ownership?", "what is beneficial ownership") == 0.0
    assert word_error_rate("one two three four", "one too three") == pytest.approx(0.5)  # 1 substitution + 1 deletion
    assert word_error_rate("a b", "a b c d") == 1.0  # two insertions
    assert normalize("İpək IŞIQ") == ["ipək", "ışıq"]
    assert word_error_rate("Faktorinq nədir?", "Faktorinq nedir?") == 0.5  # ə is not e


def test_phrase_file_is_well_formed():
    import json as _json
    from pathlib import Path

    phrases = _json.loads((Path(__file__).resolve().parent.parent / "data" / "voice_eval" / "phrases.json").read_text(encoding="utf-8"))
    assert len(phrases) == 5 and {p["language"] for p in phrases} == {"en", "az"}
    assert len({p["file"] for p in phrases}) == 5
