"""Tests for src/invoices/: schema rules, the LLM repair loop, OCR helpers, evaluation.

No API key, network or Tesseract needed: the LLM is a scripted fake and the
text-layer PDFs in data/invoices/ are read directly with pdfplumber.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

import src.api as api_module
from src.invoices.evaluate import evaluate, field_matches
from src.invoices.extract import ExtractionResult, extract_from_text, extract_invoice, parse_json_reply
from src.invoices.ocr import _deskew, estimate_skew_angle, extract_text, preprocess
from src.invoices.schema import InvoiceData

INVOICE_DIR = Path(__file__).resolve().parent.parent / "data" / "invoices"
FIELDS = ["company_name", "voen", "invoice_number", "issue_date", "total_amount", "currency"]

VALID = {
    "company_name": "Xəzər Logistika MMC",
    "voen": "1234567890",
    "invoice_number": "INV-1",
    "issue_date": "2026-03-01",
    "total_amount": "150.00",
    "currency": "AZN",
    "line_items": [
        {"description": "a", "quantity": "2", "unit_price": "50", "amount": "100"},
        {"description": "b", "quantity": "1", "unit_price": "50", "amount": "50"},
    ],
}


def _with(**changes):
    return {**VALID, **changes}


def _labels():
    return json.loads((INVOICE_DIR / "labels.json").read_text(encoding="utf-8"))


def _reply_for(label):
    return json.dumps({k: label[k] for k in FIELDS})


# --- schema / business rules -------------------------------------------------

def test_valid_invoice_passes():
    invoice = InvoiceData.model_validate(VALID)
    assert str(invoice.total_amount) == "150.00"


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"voen": "12345"}, "10 digits"),
        ({"total_amount": "0"}, "greater than 0"),
        ({"total_amount": "-5"}, "greater than 0"),
        ({"issue_date": "2999-01-01"}, "future"),
        ({"total_amount": "999.00"}, "line items sum"),
        ({"currency": "GBP"}, "currency"),
    ],
)
def test_business_rules_reject_bad_invoices(changes, message):
    with pytest.raises(ValidationError, match=message):
        InvoiceData.model_validate(_with(**changes))


def test_line_items_are_optional():
    InvoiceData.model_validate(_with(line_items=[]))


# --- LLM repair loop ----------------------------------------------------------

class ScriptedLLM:
    """Returns the given replies one after another and records the prompts it saw."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def test_parse_json_reply_handles_code_fences_and_chatter():
    assert parse_json_reply('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_reply("no json here")


# Document text that contains the values in VALID (the grounding check requires them to be printed).
DOC = "Xəzər Logistika MMC  VÖEN: 1234567890  Invoice No: INV-1  Date: 01.03.2026  TOTAL: 150.00 AZN"


def test_extraction_succeeds_first_time():
    llm = ScriptedLLM(json.dumps(VALID))
    result = extract_from_text(DOC, llm)
    assert result.invoice.voen == "1234567890"
    assert result.attempts == 1 and not result.needs_human_review


def test_extraction_repairs_after_validation_error():
    llm = ScriptedLLM(json.dumps(_with(voen="123")), json.dumps(VALID))
    result = extract_from_text(DOC, llm)
    assert result.invoice is not None and result.attempts == 2
    # the second prompt must tell the model what was wrong
    assert "10 digits" in llm.calls[1][-1]["content"]


def test_extraction_gives_up_after_two_retries_and_flags_human_review():
    bad = json.dumps(_with(voen="1"))
    llm = ScriptedLLM(bad, bad, bad)
    result = extract_from_text("text", llm)
    assert result.invoice is None
    assert result.needs_human_review is True
    assert result.attempts == 3 and len(result.errors) == 3


def test_extraction_survives_non_json_reply():
    llm = ScriptedLLM("I cannot do that", json.dumps(VALID))
    assert extract_from_text(DOC, llm).invoice is not None


# --- OCR helpers --------------------------------------------------------------

def test_text_layer_pdf_is_read_without_ocr():
    text, method = extract_text(INVOICE_DIR / "inv_01_text.pdf")
    assert method == "text_layer"
    assert "VOEN" in text and "TOTAL" in text


def test_unsupported_file_type_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text(path)


def test_extract_invoice_end_to_end_on_text_pdf():
    label = next(item for item in _labels() if item["file"] == "inv_01_text.pdf")
    result = extract_invoice(INVOICE_DIR / "inv_01_text.pdf", ScriptedLLM(_reply_for(label)))
    assert result.ocr_method == "text_layer"
    assert result.invoice.invoice_number == label["invoice_number"]


def test_deskew_reduces_the_tilt_of_a_rotated_page():
    page = Image.open(INVOICE_DIR / "inv_06_clean.png").convert("L")
    tilted = np.array(page.rotate(2.5, fillcolor=255))
    before = abs(estimate_skew_angle(tilted))
    after = abs(estimate_skew_angle(_deskew(tilted)))
    assert before > 1.5
    assert after < before / 2


def test_preprocess_returns_a_black_and_white_image():
    out = preprocess(Image.open(INVOICE_DIR / "inv_11_noisy_jpg.jpg"))
    assert set(np.unique(np.array(out))) <= {0, 255}


# --- evaluation ---------------------------------------------------------------

def test_field_matches_ignores_case_and_quotes_for_names_only():
    assert field_matches("company_name", '"Xəzər Logistika" MMC', "xəzər logistika mmc")
    assert not field_matches("voen", "123", "1234567890")
    assert field_matches("total_amount", "100.0", "100.00")
    assert not field_matches("currency", None, "AZN")


def test_evaluate_scores_a_perfect_fake_llm_at_100_percent(tmp_path):
    labels = _labels()[:2]
    for label in labels:
        shutil.copy(INVOICE_DIR / label["file"], tmp_path / label["file"])
    (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    replies = iter(_reply_for(label) for label in labels)
    report = evaluate(llm_call=lambda messages: next(replies), invoice_dir=tmp_path)
    assert report["files_evaluated"] == 2
    assert all(v == 1.0 for k, v in report["accuracy"]["all"].items() if k != "files")


# --- API endpoint -------------------------------------------------------------

client = TestClient(api_module.app)


def test_endpoint_returns_structured_invoice(monkeypatch):
    fake = ExtractionResult(invoice=InvoiceData.model_validate(VALID), attempts=1, ocr_method="text_layer")
    monkeypatch.setattr(api_module, "extract_invoice", lambda path: fake)
    response = client.post("/invoices/extract", files={"file": ("inv.pdf", b"%PDF-fake", "application/pdf")})
    assert response.status_code == 200
    assert response.json()["invoice"]["voen"] == "1234567890"


def test_endpoint_rejects_unsupported_file_type():
    response = client.post("/invoices/extract", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert response.status_code == 415


# --- Tesseract discovery ---------------------------------------------------------

def test_tesseract_command_can_be_set_from_the_environment(monkeypatch):
    import pytesseract

    from src.invoices.ocr import _configure_tesseract

    monkeypatch.setenv("TESSERACT_CMD", "C:/somewhere/tesseract.exe")
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    _configure_tesseract()
    assert pytesseract.pytesseract.tesseract_cmd == "C:/somewhere/tesseract.exe"


# --- grounding: extracted values must be printed in the document ---------------------------

def test_a_well_formed_but_invented_voen_goes_to_a_human_without_repair():
    llm = ScriptedLLM(json.dumps(_with(voen="0000000000")), json.dumps(VALID))
    result = extract_from_text(DOC.replace("1234567890", "not printed"), llm)
    assert result.invoice is None and result.needs_human_review
    assert "voen" in result.errors[-1] and result.attempts == 1  # no repair round that could invent again


def test_an_invented_total_is_caught():
    llm = ScriptedLLM(json.dumps(_with(total_amount="999.00", line_items=[])))
    result = extract_from_text(DOC, llm)
    assert result.needs_human_review and "total_amount" in result.errors[-1]


@pytest.mark.parametrize("printed", ["150.00", "150,00", "1 150,00", "150"])
def test_totals_are_found_in_common_number_formats(printed):
    from decimal import Decimal

    from src.invoices.grounding import _numbers_in

    assert Decimal(printed.replace(" ", "").replace(",", ".")) in _numbers_in(f"TOTAL: {printed} AZN")


def test_repair_prompt_forbids_guessing():
    llm = ScriptedLLM(json.dumps(_with(voen="123")), json.dumps(VALID))
    extract_from_text(DOC, llm)
    assert "do not guess" in llm.calls[1][-1]["content"]
