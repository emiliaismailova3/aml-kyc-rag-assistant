"""Mask personal data before it is stored in logs or sent to an external LLM.

Masked: e-mail addresses, Azerbaijani phone numbers, IBANs, payment card numbers (Luhn-checked),
FIN codes (only when labelled "FIN"/"FİN", because a bare 7-character code is too ambiguous) and
ID-card / passport numbers (AA1234567, AZE12345678).

NOT masked on purpose: VÖEN (a company tax ID the invoice pipeline needs and that identifies a
business, not a person), invoice numbers, dates and amounts.

Regex masking is a first line of defence, not a guarantee: names and free-text addresses are not
detected. See evals/security_eval.py for the measured recall and false-positive counts.
"""

from __future__ import annotations

import re
from collections import Counter

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# +994 50 123 45 67, (050) 123-45-67, 0501234567, 994501234567
_PHONE = re.compile(r"(?<![\w+])(?:\+?994[\s-]?|0)\(?\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_FIN = re.compile(r"(?i)\b(F[Iİ]N(?:\s*kod[u]?)?\s*[:\-]?\s*)([A-Z0-9]{7})\b")
_ID_DOC = re.compile(r"\b(?:AA\d{7}|AZE\s?\d{8})\b")


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def mask_pii(text: str) -> tuple[str, Counter]:
    """Return (masked text, Counter of what was masked)."""
    found: Counter = Counter()

    def sub(pattern: re.Pattern, label: str, value: str, keep_group: int | None = None) -> str:
        def replace(match: re.Match) -> str:
            found[label] += 1
            prefix = match.group(keep_group) if keep_group else ""
            return f"{prefix}[{label}]"

        return pattern.sub(replace, value)

    text = sub(_EMAIL, "EMAIL", text)
    text = sub(_IBAN, "IBAN", text)

    def card(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found["CARD"] += 1
            return "[CARD]"
        return match.group(0)

    text = _CARD.sub(card, text)
    text = sub(_PHONE, "PHONE", text)
    text = sub(_FIN, "FIN", text, keep_group=1)
    text = sub(_ID_DOC, "ID_DOCUMENT", text)
    return text, found


def mask(text: str) -> str:
    return mask_pii(text)[0]
