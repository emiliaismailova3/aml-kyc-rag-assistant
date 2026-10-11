"""Speech-to-text for voice questions (Whisper).

Two backends, chosen with WHISPER_PROVIDER in .env:
  local   - faster-whisper on CPU (free, private; downloads a model on first use).
            Install with: pip install -r requirements-voice.txt
  openai  - OpenAI's hosted Whisper API (needs OPENAI_API_KEY).
"""

from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".mp4"}
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # also the OpenAI API limit


class VoiceUnavailable(RuntimeError):
    """Speech-to-text cannot run (package or key missing)."""


@lru_cache(maxsize=1)
def _local_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise VoiceUnavailable(
            "Local speech-to-text needs faster-whisper: pip install -r requirements-voice.txt"
        ) from exc
    size = os.getenv("WHISPER_MODEL_SIZE", "base")
    logger.info("Loading faster-whisper model %r (first use downloads it)", size)
    # int8 on CPU keeps memory low and is fast enough for short questions.
    return WhisperModel(size, device="cpu", compute_type="int8")


def _transcribe_local(audio: bytes, filename: str, language: str | None) -> str:
    model = _local_model()
    suffix = Path(filename).suffix.lower() or ".wav"
    # The decoder reads from a file path, so the upload is written to a temporary file first.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"audio{suffix}"
        path.write_bytes(audio)
        segments, _info = model.transcribe(str(path), language=language, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()


def _transcribe_openai(audio: bytes, filename: str, language: str | None) -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise VoiceUnavailable("WHISPER_PROVIDER=openai needs OPENAI_API_KEY in .env")
    from openai import OpenAI

    kwargs = {"language": language} if language else {}
    reply = OpenAI(api_key=key).audio.transcriptions.create(model="whisper-1", file=(filename, audio), **kwargs)
    return reply.text.strip()


def transcribe(audio: bytes, filename: str, language: str | None = None) -> str:
    """Return the spoken text (empty string if nothing intelligible was said)."""
    provider = os.getenv("WHISPER_PROVIDER", "local").lower()
    if provider == "openai":
        return _transcribe_openai(audio, filename, language)
    if provider == "local":
        return _transcribe_local(audio, filename, language)
    raise VoiceUnavailable(f"Unknown WHISPER_PROVIDER={provider!r}; expected 'local' or 'openai'")
