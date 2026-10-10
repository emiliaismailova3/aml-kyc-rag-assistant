"""Generate ~15 synthetic invoices plus their ground-truth labels.

Three kinds, to exercise the different paths of src/invoices/ocr.py:
  text   - PDF with a real text layer (read directly, no OCR)
  clean  - the same kind of page rendered to a clean PNG (needs OCR)
  noisy  - a "bad scan": slightly rotated, blurred, noisy JPEG, or a PDF that
           only contains such an image (needs OCR with preprocessing)

Sellers come from data/reference/companies.json. Everything is fictional.

Usage:
    python scripts/generate_synthetic_invoices.py
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "invoices"
SEED = 7

FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]
ITEMS = [
    "Freight transport Baku - Ganja", "Office furniture set", "Consulting services (monthly)",
    "Cement M400, 50 kg bag", "Software license", "Packaging materials", "Warehouse rent",
    "Diesel fuel, litre", "Laptop computer", "Printing services",
]
ASCII_MAP = str.maketrans("əƏğĞıİöÖüÜşŞçÇ", "eEgGiIoOuUsScC")


def register_fonts() -> tuple[str, str, bool]:
    """Return (regular, bold, supports_azerbaijani). Falls back to Helvetica (ASCII only)."""
    for regular, bold in FONT_CANDIDATES:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("Body", regular))
            pdfmetrics.registerFont(TTFont("Body-Bold", bold))
            return "Body", "Body-Bold", True
    print("WARNING: no Unicode font found; falling back to Helvetica and ASCII-only names")
    return "Helvetica", "Helvetica-Bold", False


def money(value: Decimal) -> str:
    return f"{value:,.2f}"


def make_invoice(rng: random.Random, seller: dict, buyer: dict, index: int) -> dict:
    items = []
    for name in rng.sample(ITEMS, rng.randint(1, 4)):
        quantity = Decimal(rng.randint(1, 20))
        unit_price = Decimal(rng.randint(500, 90000)) / 100
        items.append(
            {"description": name, "quantity": quantity, "unit_price": unit_price,
             "amount": (quantity * unit_price).quantize(Decimal("0.01"))}
        )
    issue_date = date(2026, 1, 5) + timedelta(days=rng.randint(0, 250))
    # Some sellers appear in quotes, as on many real Azerbaijani documents.
    *words, legal = seller["name"].split()
    printed_name = f'"{" ".join(words)}" {legal}' if rng.random() < 0.3 else seller["name"]
    return {
        "company_name": printed_name,
        "reference_company_id": seller["id"],
        "voen": seller["voen"],
        "invoice_number": f"INV-2026-{rng.randint(100, 999):04d}",
        "issue_date": issue_date,
        "total_amount": sum((i["amount"] for i in items), Decimal(0)),
        "currency": rng.choice(["AZN", "AZN", "AZN", "USD", "EUR"]),
        "line_items": items,
        "buyer": buyer["name"],
    }


def draw_pdf(path: Path, inv: dict, fonts: tuple[str, str, bool]) -> None:
    regular, bold, unicode_ok = fonts
    fix = (lambda s: s) if unicode_ok else (lambda s: s.translate(ASCII_MAP))
    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 60
    pdf.setFont(bold, 18)
    pdf.drawString(50, y, "INVOICE / HESAB-FAKTURA")
    pdf.setFont(regular, 11)
    lines = [
        f"Seller: {fix(inv['company_name'])}",
        f"VOEN: {inv['voen']}",
        f"Invoice No: {inv['invoice_number']}",
        f"Date: {inv['issue_date']:%d.%m.%Y}",
        f"Buyer: {fix(inv['buyer'])}",
    ]
    for line in lines:
        y -= 22
        pdf.drawString(50, y, line)
    y -= 36
    pdf.setFont(bold, 10)
    for x, title in [(50, "Description"), (330, "Qty"), (390, "Unit price"), (480, "Amount")]:
        pdf.drawString(x, y, title)
    pdf.setFont(regular, 10)
    for item in inv["line_items"]:
        y -= 18
        pdf.drawString(50, y, item["description"])
        pdf.drawString(330, y, str(item["quantity"]))
        pdf.drawString(390, y, money(item["unit_price"]))
        pdf.drawString(480, y, money(item["amount"]))
    y -= 36
    pdf.setFont(bold, 12)
    pdf.drawString(50, y, f"TOTAL: {money(inv['total_amount'])} {inv['currency']}")
    pdf.save()


def render(pdf_path: Path, dpi: int = 200) -> Image.Image:
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[0].to_image(resolution=dpi).original.convert("RGB")


def degrade(image: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate a mediocre scan: tilt, blur and sensor noise."""
    image = image.rotate(rng.choice([-1, 1]) * rng.uniform(1.0, 3.0), expand=False, fillcolor="white")
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.1)))
    pixels = np.asarray(image).astype(np.float32)
    noise = np.random.default_rng(rng.randint(0, 10_000)).normal(0, 14, pixels.shape)
    return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))


def label_for(inv: dict, file_name: str, kind: str) -> dict:
    return {
        "file": file_name,
        "kind": kind,
        "company_name": inv["company_name"],
        "reference_company_id": inv["reference_company_id"],
        "voen": inv["voen"],
        "invoice_number": inv["invoice_number"],
        "issue_date": inv["issue_date"].isoformat(),
        "total_amount": str(inv["total_amount"]),
        "currency": inv["currency"],
        "line_item_count": len(inv["line_items"]),
    }


def main() -> None:
    rng = random.Random(SEED)
    companies = json.loads((ROOT / "data" / "reference" / "companies.json").read_text(encoding="utf-8"))
    fonts = register_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("inv_*"):
        old.unlink()

    # 5 text PDFs, 5 clean images, 3 noisy JPEGs, 2 image-only (scanned) PDFs
    plan = ["text"] * 5 + ["clean"] * 5 + ["noisy_jpg"] * 3 + ["noisy_pdf"] * 2
    sellers = rng.sample(companies, len(plan))
    labels = []
    for number, (kind, seller) in enumerate(zip(plan, sellers, strict=True), start=1):
        buyer = rng.choice([c for c in companies if c["id"] != seller["id"]])
        inv = make_invoice(rng, seller, buyer, number)
        stem = f"inv_{number:02d}_{kind}"
        if kind == "text":
            name = f"{stem}.pdf"
            draw_pdf(OUT_DIR / name, inv, fonts)
        else:
            tmp = OUT_DIR / f"{stem}.tmp.pdf"
            draw_pdf(tmp, inv, fonts)
            page = render(tmp)
            tmp.unlink()
            if kind == "clean":
                name = f"{stem}.png"
                page.save(OUT_DIR / name)
            elif kind == "noisy_jpg":
                name = f"{stem}.jpg"
                degrade(page, rng).save(OUT_DIR / name, quality=55)
            else:
                name = f"{stem}.pdf"
                degrade(page, rng).save(OUT_DIR / name, "PDF", resolution=200)
        labels.append(label_for(inv, name, kind))

    (OUT_DIR / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(labels)} invoices and labels.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
