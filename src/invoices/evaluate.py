"""Measure invoice extraction accuracy against the hand-labelled synthetic set.

For every invoice in data/invoices/labels.json the pipeline runs end to end
(read text / OCR -> LLM -> validation) and each predicted field is compared
with the ground truth. Output: per-field accuracy overall and per file kind,
saved to data/eval_results/invoice_extraction.json.

Files that need OCR are skipped (and counted as "skipped") when the Tesseract
binary is not installed, so a number is never reported for something that did
not actually run.

Usage:
    python -m src.invoices.evaluate                  # all kinds
    python -m src.invoices.evaluate --kinds text     # only PDFs with a text layer
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from src.config import PROJECT_ROOT
from src.invoices.extract import LLMCall, extract_invoice

logger = logging.getLogger(__name__)

INVOICE_DIR = PROJECT_ROOT / "data" / "invoices"
RESULT_PATH = PROJECT_ROOT / "data" / "eval_results" / "invoice_extraction.json"
FIELDS = ["company_name", "voen", "invoice_number", "issue_date", "total_amount", "currency"]


def _clean_name(name: str) -> str:
    """Compare company names ignoring case, quotes and spacing -- not the letters themselves."""
    return re.sub(r"[\"'«»“”\s]+", " ", name).strip().casefold()


def field_matches(field: str, predicted, expected: str) -> bool:
    if predicted is None:
        return False
    if field == "company_name":
        return _clean_name(str(predicted)) == _clean_name(expected)
    if field == "total_amount":
        return Decimal(str(predicted)) == Decimal(expected)
    return str(predicted) == expected


def evaluate(
    llm_call: LLMCall | None = None,
    kinds: list[str] | None = None,
    invoice_dir: Path = INVOICE_DIR,
) -> dict:
    labels = json.loads((invoice_dir / "labels.json").read_text(encoding="utf-8"))
    if kinds:
        labels = [label for label in labels if label["kind"] in kinds]

    correct: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    per_file = []
    skipped = []
    for label in labels:
        try:
            result = extract_invoice(invoice_dir / label["file"], llm_call)
        except Exception as exc:  # noqa: BLE001 - e.g. Tesseract missing: skip, never fake a score
            if type(exc).__name__ == "TesseractNotFoundError":
                skipped.append({"file": label["file"], "reason": "tesseract binary not installed"})
                continue
            raise
        invoice = result.invoice.model_dump(mode="json") if result.invoice else {}
        hits = {f: field_matches(f, invoice.get(f), label[f]) for f in FIELDS}
        for group in ("all", label["kind"]):
            totals[group] += 1
            for field, ok in hits.items():
                correct[group][field] += ok
        per_file.append(
            {"file": label["file"], "kind": label["kind"], "needs_human_review": result.needs_human_review,
             "attempts": result.attempts, "field_correct": hits}
        )

    accuracy = {
        group: {f: round(correct[group][f] / n, 3) for f in FIELDS} | {"files": n}
        for group, n in totals.items()
    }
    report = {
        "files_evaluated": totals["all"],
        "files_skipped": skipped,
        "needs_human_review": sum(f["needs_human_review"] for f in per_file),
        "accuracy": accuracy,
        "per_file": per_file,
    }
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", nargs="*", help="only these kinds: text clean noisy_jpg noisy_pdf")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = evaluate(kinds=args.kinds)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_file"}, indent=2, ensure_ascii=False))
    print(f"Saved to {RESULT_PATH}")


if __name__ == "__main__":
    _main()
