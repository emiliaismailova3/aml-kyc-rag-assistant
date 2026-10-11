"""Translation and summarization through the shared LLM client.

Both go through src.llm_client, so they get the same routing (small model for short
inputs), retries, provider fallback and call logging as every other LLM call.

Long texts are summarized map-reduce style: split into pieces, summarize each piece,
then summarize the summaries. Translation of long texts is done piece by piece too,
cut at paragraph boundaries so sentences are not split.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src.llm_client import LLMClient, get_client

GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "data" / "glossary.json"

MAX_INPUT_CHARS = 60_000   # hard limit per request; larger inputs should be sent in parts
CHUNK_CHARS = 6_000        # size of one piece for long texts

LANGUAGES = {"en": "English", "az": "Azerbaijani", "ru": "Russian", "tr": "Turkish"}

TRANSLATE_SYSTEM = (
    "You are a professional translator for financial and compliance documents. "
    "Translate the user's text into {target}. Keep names, numbers, dates, legal-entity forms "
    "and identifiers exactly as written. Do not add explanations or notes: output only the translation."
)

SUMMARIZE_SYSTEM = (
    "You summarize documents faithfully. Write the summary in {language} using at most {sentences} sentences. "
    "Use only facts present in the text; do not add opinions or outside knowledge. "
    "Keep important numbers, dates and names. Output only the summary."
)


@lru_cache(maxsize=1)
def load_glossary() -> dict[str, dict[str, str]]:
    """Fixed term translations per target language (data/glossary.json); empty if the file is missing."""
    try:
        data = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {lang: terms for lang, terms in data.items() if not lang.startswith("_")}


def glossary_prompt(target: str) -> str:
    terms = load_glossary().get(target.lower(), {})
    if not terms:
        return ""
    lines = "\n".join(f"- {source} -> {translation}" for source, translation in terms.items())
    return f"\nAlways translate these terms exactly as listed:\n{lines}"


class TextToolError(ValueError):
    """The request itself is invalid (unknown language, empty or too large input)."""


def language_name(code: str) -> str:
    try:
        return LANGUAGES[code.lower()]
    except KeyError:
        raise TextToolError(f"Unsupported language {code!r}. Use one of: {', '.join(sorted(LANGUAGES))}.") from None


def _check_text(text: str) -> str:
    text = text.strip()
    if not text:
        raise TextToolError("The text is empty.")
    if len(text) > MAX_INPUT_CHARS:
        raise TextToolError(f"The text is longer than {MAX_INPUT_CHARS} characters; send it in parts.")
    return text


def split_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split at paragraph (then sentence, then hard) boundaries into pieces of at most `limit` characters."""
    pieces: list[str] = []
    current = ""
    for paragraph in text.split("\n"):
        while len(paragraph) > limit:  # one huge paragraph: cut at the last sentence end before the limit
            cut = paragraph.rfind(". ", 0, limit)
            cut = cut + 1 if cut > 0 else limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(paragraph[:cut])
            paragraph = paragraph[cut:].lstrip()
        if len(current) + len(paragraph) + 1 > limit and current:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return [p for p in pieces if p.strip()]


def translate(text: str, target: str, client: LLMClient | None = None) -> str:
    system = TRANSLATE_SYSTEM.format(target=language_name(target)) + glossary_prompt(target)
    client = client or get_client()
    parts = []
    for piece in split_text(_check_text(text)):
        reply = client.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": piece}], task="translate"
        )
        parts.append(reply.text.strip())
    return "\n".join(parts)


def summarize(text: str, language: str = "en", max_sentences: int = 5, client: LLMClient | None = None) -> str:
    if not 1 <= max_sentences <= 20:
        raise TextToolError("max_sentences must be between 1 and 20.")
    system = SUMMARIZE_SYSTEM.format(language=language_name(language), sentences=max_sentences)
    client = client or get_client()
    text = _check_text(text)

    def run(piece: str) -> str:
        reply = client.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": piece}], task="summarize"
        )
        return reply.text.strip()

    pieces = split_text(text)
    if len(pieces) == 1:
        return run(pieces[0])
    partial = [run(piece) for piece in pieces]          # map: one summary per piece
    combined = "\n".join(partial)
    return run(combined) if len(combined) <= MAX_INPUT_CHARS else combined  # reduce
