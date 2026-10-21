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

from src.invoices.grounding import ungrounded_fields
from src.invoices.ocr import extract_text
from src.invoices.schema import InvoiceData
from src.llm_client import LLMUnavailable
from src.security.guard import mask_prompts_enabled
from src.security.pii import mask

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

Never invent values. If a field is unreadable, give your best reading of the text.
If a field is not printed in the document at all (for example there is no VÖEN, or the currency is
not AZN, USD or EUR), use null for it. Never fill a field with a different number, a placeholder
such as 0000000000, or a currency that is not printed."""

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
    from src.llm_client import get_client

    # More than [system, user] means the first answer failed validation: use the main model.
    return get_client().complete(messages, task="invoice_extraction", retry=len(messages) > 2).text


def extract_from_text(text: str, llm_call: LLMCall | None = None) -> ExtractionResult:
    llm_call = llm_call or _default_llm_call
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"INVOICE TEXT:\n{text}"},
    ]
    errors: list[str] = []
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            reply = llm_call(messages)
        except LLMUnavailable as exc:
            # Every provider is down: do not lose the document, hand it to a person.
            return ExtractionResult(
                needs_human_review=True, errors=errors + [str(exc)], attempts=attempt
            )
        try:
            invoice = InvoiceData.model_validate(parse_json_reply(reply))
        except (ValidationError, ValueError) as exc:
            error = _short_error(exc)
            errors.append(error)
            logger.info("Invoice extraction attempt %d failed validation: %s", attempt, error)
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That JSON is invalid: {error}\nFix it and reply with the corrected JSON only. "
                    "If the correct value is not in the invoice text, use null: do not guess or make one up.",
                },
            ]
            continue
        missing = ungrounded_fields(invoice, text)
        if not missing:
            return ExtractionResult(invoice=invoice, attempts=attempt)
        # Well-formed but not taken from the document. Asking the model to "fix" it only invites
        # another invented value, so it goes straight to a person (see src/invoices/grounding.py).
        return ExtractionResult(
            needs_human_review=True, attempts=attempt,
            errors=errors + [f"value(s) not found in the document: {', '.join(missing)}"],
        )
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
    if mask_prompts_enabled():
        # Phones, e-mails, IBANs and card numbers are not needed for the extracted fields; VÖEN,
        # invoice numbers, dates and amounts are left untouched (see src/security/pii.py).
        text = mask(text)
    result = extract_from_text(text, llm_call)
    result.ocr_method = method
    return result
