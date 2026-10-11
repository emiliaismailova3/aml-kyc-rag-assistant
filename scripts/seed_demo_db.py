"""Fill the database with demo data so the agent's SQL tool has something to query.

Creates the tables, loads the 50 reference companies, and inserts the 15 synthetic
invoices from data/invoices/labels.json. The invoice rows come from the GROUND-TRUTH
labels (no LLM or OCR involved), so this is demo data, not extraction output.

Works on SQLite by default; set DATABASE_URL to use PostgreSQL.

Usage:
    python -m scripts.seed_demo_db
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

from sqlalchemy import delete

from src.config import PROJECT_ROOT
from src.db import get_engine, init_db, invoices, save_invoice, seed_companies


def main() -> None:
    engine = init_db(get_engine())
    added = seed_companies(engine)
    labels = json.loads((PROJECT_ROOT / "data" / "invoices" / "labels.json").read_text(encoding="utf-8"))
    with engine.begin() as conn:
        conn.execute(delete(invoices))  # re-running replaces the demo invoices instead of duplicating them
    for label in labels:
        save_invoice(
            {
                "file_name": label["file"],
                "company_id": label["reference_company_id"],
                "company_name": label["company_name"],
                "voen": label["voen"],
                "invoice_number": label["invoice_number"],
                "issue_date": date.fromisoformat(label["issue_date"]),
                "total_amount": Decimal(label["total_amount"]),
                "currency": label["currency"],
                "needs_human_review": False,
            },
            engine,
        )
    print(f"Seeded {added} new companies and {len(labels)} demo invoices into {engine.url.render_as_string(hide_password=True)}")


if __name__ == "__main__":
    main()
