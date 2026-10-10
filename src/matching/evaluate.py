"""Precision / recall of company matching on the labeled pairs.

A pair counts as "accepted" when the matcher returns a company without asking
for human review. Then:
  precision = accepted and correct / all accepted      (are the automatic matches right?)
  recall    = accepted and correct / all pairs that have a true company
Pairs sent to human review are reported separately: they are neither wrong nor automatic.

Usage:
    python -m src.matching.evaluate
"""

from __future__ import annotations

import json

from src.config import PROJECT_ROOT
from src.matching import matcher as matcher_module
from src.matching.matcher import CompanyMatcher, MatchResult, load_companies

PAIRS_PATH = PROJECT_ROOT / "data" / "reference" / "matching_pairs.json"
RESULT_PATH = PROJECT_ROOT / "data" / "eval_results" / "matching.json"


def score_predictions(pairs: list[dict], results: list[MatchResult]) -> dict:
    accepted = [(p, r) for p, r in zip(pairs, results, strict=True) if r.company_id and not r.needs_human_review]
    true_positive = sum(r.company_id == p["expected_id"] for p, r in accepted)
    positives = sum(p["expected_id"] is not None for p in pairs)
    return {
        "pairs": len(pairs),
        "accepted": len(accepted),
        "sent_to_human_review": sum(r.needs_human_review for r in results),
        "true_positives": true_positive,
        "false_positives": len(accepted) - true_positive,
        "precision": round(true_positive / len(accepted), 3) if accepted else None,
        "recall": round(true_positive / positives, 3) if positives else None,
    }


def evaluate(matcher: CompanyMatcher, pairs: list[dict]) -> dict:
    results = [matcher.match(p["query"]) for p in pairs]
    report = score_predictions(pairs, results)
    by_kind: dict[str, dict] = {}
    for kind in sorted({p["kind"] for p in pairs}):
        idx = [i for i, p in enumerate(pairs) if p["kind"] == kind]
        by_kind[kind] = score_predictions([pairs[i] for i in idx], [results[i] for i in idx])
    report["by_kind"] = by_kind
    report["errors"] = [
        {"query": p["query"], "expected_id": p["expected_id"], "got": r.model_dump()}
        for p, r in zip(pairs, results, strict=True)
        if (r.company_id != p["expected_id"]) or r.needs_human_review
    ]
    return report


def _main() -> None:
    pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    report = {
        "thresholds": {
            "fuzzy_accept": matcher_module.FUZZY_ACCEPT,
            "embedding_accept": matcher_module.EMBEDDING_ACCEPT,
            "review": matcher_module.REVIEW_THRESHOLD,
        },
        **evaluate(CompanyMatcher(load_companies()), pairs),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("errors", "by_kind")}, indent=2))
    for kind, stats in report["by_kind"].items():
        print(f"  {kind:18s} accepted {stats['accepted']}/{stats['pairs']}  precision {stats['precision']}")
    print(f"{len(report['errors'])} pairs were wrong or sent to review (see {RESULT_PATH})")


if __name__ == "__main__":
    _main()
