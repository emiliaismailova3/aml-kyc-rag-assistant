"""Turn raw HTML / PDF bytes into clean, normalized text and a stable content hash."""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata

import pdfplumber
from bs4 import BeautifulSoup

# Page furniture that is not content.
_NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "sup"]


def html_to_text(html: bytes | str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    return root.get_text(separator="\n")


def pdf_to_text(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def normalize_text(text: str) -> str:
    """Unicode NFKC (e.g. non-breaking spaces, ligatures), unified whitespace, no blank runs."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")  # soft hyphens
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n"))
    # Keep paragraph breaks (one blank line) because the chunker splits on them first.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_text(body: bytes, content_type: str) -> str:
    raw = pdf_to_text(body) if content_type == "application/pdf" else html_to_text(body)
    return normalize_text(raw)


def content_hash(text: str) -> str:
    """Hash of the text ignoring case and whitespace differences, so a page that only
    changed formatting (or a trailing space) counts as the same document."""
    canonical = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
