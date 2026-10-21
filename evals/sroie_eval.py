"""Invoice pipeline on REAL scanned receipts: 30 from the ICDAR 2019 SROIE test set (CC BY 4.0,
Malaysian shop receipts, English). Download them first (see evals/README or the session notes)
into data/external/sroie/ (git-ignored; not redistributed).

What it measures, per receipt:
  - field accuracy of the model's FIRST JSON reply (before validation): seller company, date, total
  - what the validation layer did: SROIE receipts have no VÖEN and use MYR, so the correct outcome is
    needs_human_review. A receipt that ends up ACCEPTED would mean the model invented a VÖEN or a
    currency during the repair loop -- the dangerous failure this test looks for.

Real LLM calls (small model, about 3 calls per receipt because of the repair loop).

    python -m evals.sroie_eval
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.invoices.extract import _default_llm_call, extract_from_text, parse_json_reply
from src.invoices.ocr import extract_text

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "external" / "sroie"
OUT = ROOT / "evals" / "results" / "sroie_eval.json"
PAUSE_SECONDS = 15.0
_LEGAL = {"sdn", "bhd", "sdn.", "bhd.", "berhad", "enterprise", "trading", "s/b"}


def norm_company(name) -> str:
    words = re.sub(r"[^a-z0-9 ]", " ", str(name or "").lower()).split()
    return " ".join(w for w in words if w not in _LEGAL)


def parse_date(value) -> date | None:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%d.%m.%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_amount(value) -> Decimal | None:
    try:
        return Decimal(re.sub(r"[^\d.\-]", "", str(value)))
    except (InvalidOperation, TypeError):
        return None


def main() -> None:
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    rows = []
    for label in labels:
        time.sleep(PAUSE_SECONDS)
        text, method = extract_text(DATA / label["file"])
        first_reply: dict = {}

        def recording_call(messages, _seen=first_reply):
            reply = _default_llm_call(messages)
            if not _seen:
                _seen["text"] = reply
            return reply

        result = extract_from_text(text, recording_call)
        try:
            first = parse_json_reply(first_reply.get("text", ""))
        except (ValueError, json.JSONDecodeError):
            first = {}
        row = {
            "file": label["file"], "ocr_chars": len(text), "attempts": result.attempts,
            "outcome": "needs_human_review" if result.needs_human_review else "accepted",
            "company_ok": bool(norm_company(first.get("company_name"))) and norm_company(first.get("company_name")) == norm_company(label["company"]),
            "date_ok": parse_date(first.get("issue_date")) is not None and parse_date(first.get("issue_date")) == parse_date(label["date"]),
            "total_ok": parse_amount(first.get("total_amount")) is not None and parse_amount(first.get("total_amount")) == parse_amount(label["total"]),
            "first_voen": first.get("voen"), "first_currency": first.get("currency"),
        }
        if result.invoice is not None:
            row["accepted_voen"] = result.invoice.voen
            row["accepted_currency"] = result.invoice.currency
        rows.append(row)
        print({k: row[k] for k in ("file", "outcome", "company_ok", "date_ok", "total_ok")}, flush=True)
    n = len(rows)
    report = {
        "dataset": "ICDAR 2019 SROIE test split, first 30 receipts (real scans, CC BY 4.0)",
        "receipts": n,
        "first_reply_accuracy": {f: round(sum(r[f"{f}_ok"] for r in rows) / n, 3) for f in ("company", "date", "total")},
        "sent_to_human_review": sum(r["outcome"] == "needs_human_review" for r in rows),
        "accepted": sum(r["outcome"] == "accepted" for r in rows),
        "accepted_files": [r["file"] for r in rows if r["outcome"] == "accepted"],
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
