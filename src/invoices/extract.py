"""LLM extraction of invoice fields, with validation and a repair loop.

Flow: text from ocr.py -> LLM returns JSON -> Pydantic validates it.
If validation fails, the error message is sent back to the LLM and it tries
again (at most MAX_RETRIES times). If it still fails, the result is marked
needs_human_review instead of returning data we cannot trust.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.invoices.ocr import extract_text
from src.invoices.schema import InvoiceData

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

SYSTEM_PROMPT = """You extract structured data from invoices (Azerbaijani business documents; \
the text may come from OCR and contain noise). Reply with ONE JSON object and nothing else.

Fields:
- company_name: the SELLER / supplier company issuing the invoice (not the buyer), exactly as printed
- voen: the seller's 10-digit tax number (VOEN / VÖEN)
- invoice_number: the invoice number as printed
- issue_date: ISO format YYYY-MM-DD
- total_amount: the final total to pay, as a number (use a dot as decimal separator, no thousands separators)
- currency: one of AZN, USD, EUR
- line_items: list of {"description", "quantity", "unit_price", "amount"}; use [] if there is no item table

Never invent values. If a field is unreadable, give your best reading of the text."""

# (messages) -> reply text. Messages are {"role": ..., "content": ...} dicts.
LLMCall = Callable[[list[dict]], str]


class ExtractionResult(BaseModel):
    invoice: InvoiceData | None = None
    needs_human_review: bool = False
    errors: list[str] = []
    ocr_method: str | None = None
    attempts: int = 0


def parse_json_reply(reply: str) -> dict:
    """The model sometimes wraps the JSON in ```json fences or adds a sentence around it."""
    match = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    if not match:
        raise ValueError("the reply contains no JSON object")
    return json.loads(match.group(0))


def _default_llm_call(messages: list[dict]) -> str:
    from src.rag import get_llm

    return get_llm().invoke([(m["role"], m["content"]) for m in messages]).content


def extract_from_text(text: str, llm_call: LLMCall | None = None) -> ExtractionResult:
    llm_call = llm_call or _default_llm_call
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"INVOICE TEXT:\n{text}"},
    ]
    errors: list[str] = []
    for attempt in range(1, MAX_RETRIES + 2):
        reply = llm_call(messages)
        try:
            invoice = InvoiceData.model_validate(parse_json_reply(reply))
            return ExtractionResult(invoice=invoice, attempts=attempt)
        except (ValidationError, ValueError) as exc:
            error = _short_error(exc)
            errors.append(error)
            logger.info("Invoice extraction attempt %d failed validation: %s", attempt, error)
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That JSON is invalid: {error}\nFix it and reply with the corrected JSON only.",
                },
            ]
    return ExtractionResult(needs_human_review=True, errors=errors, attempts=MAX_RETRIES + 1)


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'invoice'}: {e['msg']}" for e in exc.errors()
        )
    return str(exc)


def extract_invoice(path: str | Path, llm_call: LLMCall | None = None) -> ExtractionResult:
    """Full pipeline for one invoice file (PDF or image)."""
    text, method = extract_text(path)
    if not text.strip():
        return ExtractionResult(
            needs_human_review=True, errors=["no text could be read from the file"], ocr_method=method
        )
    result = extract_from_text(text, llm_call)
    result.ocr_method = method
    return result
