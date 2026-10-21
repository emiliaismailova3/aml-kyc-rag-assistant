"""Offline security evaluation (no LLM calls).

Prompt injection: detection rate on data/security/injection_cases.json attacks, by kind
(direct / paraphrase / indirect), and false-positive rate on benign questions: the 15 gold
questions, the 16 agent scenarios and 20 hand-written tricky benign questions.

PII masking: per-entity recall and precision on data/security/pii_cases.json, plus how many
must-keep substrings (VÖEN, invoice numbers, amounts, dates) were damaged. Also checks that masking
the 15 invoice texts leaves every ground-truth field value intact.

    python -m evals.security_eval
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from src.security.injection import detect_injection
from src.security.pii import mask_pii

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evals" / "results" / "security_eval.json"


def injection_report() -> dict:
    cases = json.loads((ROOT / "data" / "security" / "injection_cases.json").read_text(encoding="utf-8"))
    by_kind: dict[str, list[bool]] = defaultdict(list)
    missed = []
    for a in cases["attacks"]:
        hit = detect_injection(a["text"]).flagged
        by_kind[a["kind"]].append(hit)
        if not hit:
            missed.append(a["id"])
    gold = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))["questions"]
    scenarios = json.loads((ROOT / "data" / "agent_test_scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    benign = ([("gold:" + q["id"], q["question"]) for q in gold] + [("scenario:" + s["id"], s["question"]) for s in scenarios]
              + [("extra:" + b["id"], b["text"]) for b in cases["benign_extra"]])
    false_positives = [(i, detect_injection(t).rules) for i, t in benign if detect_injection(t).flagged]
    total_attacks = sum(len(v) for v in by_kind.values())
    return {
        "attacks": total_attacks,
        "detected": total_attacks - len(missed),
        "detection_rate": round((total_attacks - len(missed)) / total_attacks, 3),
        "by_kind": {k: f"{sum(v)}/{len(v)}" for k, v in by_kind.items()},
        "missed": missed,
        "benign": len(benign),
        "false_positives": len(false_positives),
        "false_positive_rate": round(len(false_positives) / len(benign), 3),
        "false_positive_ids": false_positives,
    }


def pii_report() -> dict:
    cases = json.loads((ROOT / "data" / "security" / "pii_cases.json").read_text(encoding="utf-8"))["cases"]
    expected_total, found_total, correct_total = Counter(), Counter(), Counter()
    damaged, wrong = [], []
    for c in cases:
        masked, found = mask_pii(c["text"])
        exp = Counter(c["expected"])
        expected_total.update(exp)
        found_total.update(found)
        correct_total.update(exp & found)
        if found != exp:
            wrong.append({"id": c["id"], "expected": dict(exp), "found": dict(found), "masked": masked})
        for keep in c.get("keep", []):
            if keep not in masked:
                damaged.append({"id": c["id"], "lost": keep, "masked": masked})
    entities = sorted(set(expected_total) | set(found_total))
    per_entity = {e: {"expected": expected_total[e], "found": found_total[e], "correct": correct_total[e]} for e in entities}
    n_exp, n_found, n_ok = sum(expected_total.values()), sum(found_total.values()), sum(correct_total.values())
    return {
        "cases": len(cases), "entities_expected": n_exp,
        "recall": round(n_ok / n_exp, 3), "precision": round(n_ok / n_found, 3) if n_found else None,
        "per_entity": per_entity, "cases_not_exact": wrong, "must_keep_values_damaged": damaged,
    }


def invoice_field_survival() -> dict:
    from src.invoices.ocr import extract_text

    labels = json.loads((ROOT / "data" / "invoices" / "labels.json").read_text(encoding="utf-8"))
    broken = []
    for label in labels:
        if label["kind"] != "text":  # text-layer PDFs only: no OCR dependency, exact ground-truth strings
            continue
        text, _ = extract_text(ROOT / "data" / "invoices" / label["file"])
        masked, found = mask_pii(text)
        for field in ("voen", "invoice_number"):
            if str(label[field]) in text and str(label[field]) not in masked:
                broken.append({"file": label["file"], "field": field, "masked_entities": dict(found)})
    return {"text_invoices_checked": sum(1 for x in labels if x["kind"] == "text"), "fields_destroyed": broken}


def main() -> None:
    report = {"prompt_injection": injection_report(), "pii": pii_report(), "invoice_fields_after_masking": invoice_field_survival()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
