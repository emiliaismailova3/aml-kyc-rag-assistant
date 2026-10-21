"""Effect of the reranker on retrieval, without any LLM call.

For each gold question, compare the chunks the RAG pipeline and the agent would see with and
without reranking: document-level hit rate at k, and the reranker's added latency per query.

    python -m evals.rerank_effect
"""

from __future__ import annotations

import ast
import json
import statistics
import time
from pathlib import Path

from src.rerank import candidate_count, rerank
from src.vectorstore import load_vectorstore

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evals" / "results" / "rerank_effect.json"


def main() -> None:
    questions = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))["questions"]
    store = load_vectorstore()
    rerank("warm up", store.similarity_search("warm up", k=2), 1)  # load the model outside the timing
    rows, rerank_ms = [], []
    for q in questions:
        expected = set(ast.literal_eval(q["source_documents"]) if isinstance(q["source_documents"], str) else q["source_documents"])
        candidates = store.similarity_search(q["question"], k=candidate_count())
        t = time.perf_counter()
        reranked = rerank(q["question"], candidates, k=8)
        rerank_ms.append((time.perf_counter() - t) * 1000)
        row = {"id": q["id"]}
        for label, docs in (("vector", candidates), ("reranked", reranked)):
            names = [Path(d.metadata.get("source", "")).name for d in docs]
            for k in (1, 4):
                row[f"{label}_hit@{k}"] = any(n in expected for n in names[:k])
            row[f"{label}_top4_chunks"] = [f"{Path(d.metadata.get('source', '')).name}:p{d.metadata.get('page')}" for d in docs[:4]]
        row["top4_changed"] = row["vector_top4_chunks"] != row["reranked_top4_chunks"]
        rows.append(row)
    n = len(rows)
    report = {
        "questions": n, "candidates": candidate_count(),
        **{f"{lab}_hit_rate@{k}": round(sum(r[f"{lab}_hit@{k}"] for r in rows) / n, 3) for lab in ("vector", "reranked") for k in (1, 4)},
        "questions_with_different_top4": sum(r["top4_changed"] for r in rows),
        "rerank_latency_ms": {"p50": round(statistics.median(rerank_ms), 1), "max": round(max(rerank_ms), 1)},
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
