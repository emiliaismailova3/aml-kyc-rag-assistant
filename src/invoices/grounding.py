"""Grounding check: the key values the LLM extracted must actually appear in the document text.

Why: on 30 real scanned receipts (ICDAR 2019 SROIE, evals/sroie_eval.py) the model, pushed by the
validation repair loop, invented a 10-digit VÖEN (e.g. "0000000000") and an allowed currency for
documents that have neither, and 6 of 30 receipts were accepted with made-up data. Pydantic rules
check the *shape* of a value; this checks that the value came from the document.

Checked: VÖEN (its digits must appear), invoice number (letters and digits must appear), total
amount (must equal some number printed in the text). Dates are not checked: they are printed in
too many formats to compare reliably.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from src.invoices.schema import InvoiceData

_NUMBER = re.compile(r"\d[\d\s.,']*\d|\d")


def _compact(value: str) -> str:
    return re.sub(r"[^0-9a-z]", "", value.lower())


def _numbers_in(text: str) -> set[Decimal]:
    """Every number printed in the text, reading both 1,234.56 and 1.234,56 / 1 234,56 styles."""
    found: set[Decimal] = set()
    for raw in _NUMBER.findall(text):
        token = re.sub(r"[\s']", "", raw).strip(".,")
        candidates = {token.replace(",", ""), token.replace(".", "").replace(",", ".")}
        if token.count(",") == 1 and "." not in token:
            candidates.add(token.replace(",", "."))
        for candidate in candidates:
            try:
                found.add(Decimal(candidate))
            except InvalidOperation:
                continue
    return found


def ungrounded_fields(invoice: InvoiceData, text: str) -> list[str]:
    """Names of the fields whose value does not appear in the document text."""
    missing = []
    digits_only = re.sub(r"\D", "", text)
    compact_text = _compact(text)
    if invoice.voen not in digits_only or len(set(invoice.voen)) == 1:  # "0000000000" is a placeholder
        missing.append("voen")
    if _compact(invoice.invoice_number) not in compact_text:
        missing.append("invoice_number")
    if invoice.total_amount not in _numbers_in(text):
        missing.append("total_amount")
    return missing
