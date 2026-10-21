"""Retrieval hit rate on the 15 gold questions, document level. No LLM calls, no cost.

A question is a "hit at k" when at least one of the top-k retrieved chunks comes from a PDF
listed in the question's `source_documents`. This is DOCUMENT-level, not passage-level:
it shows whether the right document is found, not whether the exact answer passage is.

    python -m evals.retrieval_hit_rate
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from src.vectorstore import load_vectorstore

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evals" / "results" / "retrieval_hit_rate.json"
KS = (1, 4, 8)


def main() -> None:
    questions = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))["questions"]
    store = load_vectorstore()
    rows = []
    for q in questions:
        expected = q["source_documents"]
        expected = set(ast.literal_eval(expected) if isinstance(expected, str) else expected)
        docs = store.similarity_search(q["question"], k=max(KS))
        sources = [Path(d.metadata.get("source", "")).name for d in docs]
        first = next((i + 1 for i, s in enumerate(sources) if s in expected), None)
        rows.append({"id": q["id"], "expected": sorted(expected), "first_hit_rank": first,
                     **{f"hit@{k}": first is not None and first <= k for k in KS}})
    report = {"questions": len(rows), **{f"hit_rate@{k}": round(sum(r[f"hit@{k}"] for r in rows) / len(rows), 3) for k in KS},
              "level": "document", "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "rows"})
    print("misses at 8:", [r["id"] for r in rows if not r["hit@8"]])


if __name__ == "__main__":
    main()
