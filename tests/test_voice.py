"""Tests for voice questions (src/voice.py and POST /ask_voice).

Whisper itself is replaced by fakes, so no model download, microphone or API key is needed.
"""

import types

import pytest
from fastapi.testclient import TestClient

import src.api as api_module
import src.voice as voice

client = TestClient(api_module.app)
WAV = ("question.wav", b"RIFF....WAVEfmt ", "audio/wav")


class FakeAgent:
    def __init__(self):
        self.questions = []

    def answer(self, question):
        self.questions.append(question)
        return {"question": question, "answer": "It is 16650.", "tool_calls": [], "sources": [], "contexts": []}


@pytest.fixture
def fake_agent(monkeypatch):
    agent = FakeAgent()
    monkeypatch.setattr(api_module, "_agent_pipeline", agent)
    monkeypatch.setattr(api_module, "log_request", lambda **kwargs: None)
    return agent


# --- endpoint ----------------------------------------------------------------------------

def test_spoken_question_is_transcribed_and_sent_to_the_agent(monkeypatch, fake_agent):
    monkeypatch.setattr(api_module, "transcribe", lambda audio, name, language: "What is 450 times 37?")
    response = client.post("/ask_voice", files={"file": WAV})
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "What is 450 times 37?" and body["answer"] == "It is 16650."
    assert fake_agent.questions == ["What is 450 times 37?"]


def test_language_hint_is_passed_to_whisper(monkeypatch, fake_agent):
    seen = {}
    monkeypatch.setattr(api_module, "transcribe", lambda audio, name, language: seen.update(language=language) or "salam")
    client.post("/ask_voice", files={"file": WAV}, data={"language": "az"})
    assert seen["language"] == "az"


def test_unsupported_file_type_is_rejected():
    response = client.post("/ask_voice", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_silence_returns_a_clear_422(monkeypatch, fake_agent):
    monkeypatch.setattr(api_module, "transcribe", lambda audio, name, language: "")
    response = client.post("/ask_voice", files={"file": WAV})
    assert response.status_code == 422 and not fake_agent.questions


def test_oversized_audio_is_rejected(monkeypatch):
    monkeypatch.setattr(api_module, "MAX_AUDIO_BYTES", 10)
    assert client.post("/ask_voice", files={"file": ("big.wav", b"x" * 11, "audio/wav")}).status_code == 413


def test_missing_whisper_setup_returns_503(monkeypatch):
    def unavailable(audio, name, language):
        raise voice.VoiceUnavailable("install faster-whisper")

    monkeypatch.setattr(api_module, "transcribe", unavailable)
    response = client.post("/ask_voice", files={"file": WAV})
    assert response.status_code == 503 and "faster-whisper" in response.json()["detail"]


# --- backends --------------------------------------------------------------------------------

def test_local_backend_joins_segments_and_passes_options(monkeypatch):
    calls = {}

    class FakeModel:
        def transcribe(self, path, language, vad_filter):
            calls.update(language=language, vad=vad_filter, existed=__import__("os").path.exists(path))
            return iter([types.SimpleNamespace(text=" Hello "), types.SimpleNamespace(text="world ")]), None

    monkeypatch.setenv("WHISPER_PROVIDER", "local")
    monkeypatch.setattr(voice, "_local_model", lambda: FakeModel())
    assert voice.transcribe(b"audio", "q.mp3", language="en") == "Hello world"
    assert calls == {"language": "en", "vad": True, "existed": True}


def test_openai_backend_needs_a_key(monkeypatch):
    monkeypatch.setenv("WHISPER_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(voice.VoiceUnavailable, match="OPENAI_API_KEY"):
        voice.transcribe(b"audio", "q.wav")


def test_openai_backend_calls_the_transcription_api(monkeypatch):
    monkeypatch.setenv("WHISPER_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen = {}

    class FakeOpenAI:
        def __init__(self, api_key):
            self.audio = types.SimpleNamespace(
                transcriptions=types.SimpleNamespace(create=lambda **kwargs: seen.update(kwargs) or types.SimpleNamespace(text=" hi "))
            )

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    assert voice.transcribe(b"audio", "q.wav", language="az") == "hi"
    assert seen["model"] == "whisper-1" and seen["file"] == ("q.wav", b"audio") and seen["language"] == "az"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("WHISPER_PROVIDER", "carrier-pigeon")
    with pytest.raises(voice.VoiceUnavailable, match="Unknown WHISPER_PROVIDER"):
        voice.transcribe(b"audio", "q.wav")


def test_local_backend_explains_how_to_install_faster_whisper(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_faster_whisper(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    voice._local_model.cache_clear()
    monkeypatch.setattr(builtins, "__import__", no_faster_whisper)
    with pytest.raises(voice.VoiceUnavailable, match="requirements-voice.txt"):
        voice._local_model()
