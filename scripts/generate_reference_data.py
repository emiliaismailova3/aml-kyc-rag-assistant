"""Generate the fictional reference data used by the invoice and matching modules.

Writes two files (deterministic: the same seed always gives the same output):
  data/reference/companies.json       ~50 fictional Azerbaijani companies (name + VOEN)
  data/reference/matching_pairs.json  labeled (messy name -> expected company id) pairs

All companies are invented. VOENs are random 10-digit numbers, not real taxpayers.

Usage:
    python scripts/generate_reference_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"

FIRST_WORDS = [
    "Xəzər", "Bakı", "Qafqaz", "Günəş", "Şəki", "Gəncə", "Ülkər", "Aygün", "Çinar", "Dəniz",
    "Ədalət", "Füzuli", "Göyçay", "Həyat", "İpək", "Kəpəz", "Lənkəran", "Mingəçevir", "Naxçıvan", "Odlar",
]
SECTORS = [
    "Logistika", "Tikinti", "Ticarət", "Qida", "Texnologiya", "Enerji", "Tekstil", "Neft Servis",
    "Nəqliyyat", "Aqrar", "Dərman", "Mebel", "Kimya", "Telekom", "Konsaltinq",
]
LEGAL_FORMS = ["MMC", "MMC", "MMC", "QSC", "ASC", "LLC"]

# Azerbaijani letters -> what an English-only keyboard/OCR typically produces.
ASCII_MAP = str.maketrans("əƏğĞıİöÖüÜşŞçÇ", "eEgGiIoOuUsScC")


def make_companies(rng: random.Random, count: int = 50) -> list[dict]:
    combos = [(a, b) for a in FIRST_WORDS for b in SECTORS]
    rng.shuffle(combos)
    companies = []
    used_voens: set[str] = set()
    for index, (first, sector) in enumerate(combos[:count], start=1):
        voen = str(rng.randint(10**9, 10**10 - 1))
        while voen in used_voens:
            voen = str(rng.randint(10**9, 10**10 - 1))
        used_voens.add(voen)
        companies.append(
            {
                "id": index,
                "name": f"{first} {sector} {rng.choice(LEGAL_FORMS)}",
                "voen": voen,
            }
        )
    return companies


def typo(text: str, rng: random.Random) -> str:
    """Swap two neighbouring letters in the longest word (a typical typing/OCR slip)."""
    words = text.split()
    index = max(range(len(words)), key=lambda i: len(words[i]))
    word = words[index]
    if len(word) > 3:
        pos = rng.randrange(1, len(word) - 2)
        word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
    words[index] = word
    return " ".join(words)


def make_pairs(companies: list[dict], rng: random.Random) -> list[dict]:
    pairs = []
    sample = rng.sample(companies, 30)
    for company in sample[:6]:
        pairs.append({"query": company["name"], "expected_id": company["id"], "kind": "exact"})
    for company in sample[6:12]:
        pairs.append({"query": company["name"].upper(), "expected_id": company["id"], "kind": "uppercase"})
    for company in sample[12:18]:
        pairs.append(
            {"query": company["name"].translate(ASCII_MAP), "expected_id": company["id"], "kind": "no_diacritics"}
        )
    for company in sample[18:22]:
        *words, legal = company["name"].split()
        pairs.append({"query": " ".join(words), "expected_id": company["id"], "kind": "no_legal_form"})
    for company in sample[22:25]:
        *words, legal = company["name"].split()
        pairs.append(
            {"query": f'"{" ".join(words)}" {"LLC" if legal != "LLC" else "MMC"}', "expected_id": company["id"],
             "kind": "quotes_other_form"}
        )
    for company in sample[25:30]:
        pairs.append({"query": typo(company["name"], rng), "expected_id": company["id"], "kind": "typo"})
    # Companies that are NOT in the reference table: the matcher must not invent a match.
    for name in [
        "Sunrise Trading Ltd", "Karabakh Gold Mining ASC", "Alpha Beta Consulting", "Zeytun Bakery MMC",
        "Global Freight Partners", "Mavi Dalğa Turizm MMC", "Nova Pharma LLC", "Atlas Steel QSC",
        "Pristine Cleaning Services", "Orion Software Hub",
    ]:
        pairs.append({"query": name, "expected_id": None, "kind": "not_in_table"})
    return pairs


def main() -> None:
    rng = random.Random(SEED)
    companies = make_companies(rng)
    pairs = make_pairs(companies, rng)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "companies.json").write_text(json.dumps(companies, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "matching_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(companies)} companies and {len(pairs)} labeled pairs to {OUT_DIR}")


if __name__ == "__main__":
    main()
