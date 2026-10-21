"""A/B test of the reranker on outcomes that need no LLM judge.

  rag_k4   - base RAG with 4 chunks, reranker off vs on: how many of the 15 gold questions are refused
             ("I don't know ...")? Refusal detection is the deterministic src.rag.is_refusal.
  agent    - the agent (4 chunks per search) with the reranker on: how many gold questions end in
             the step limit (GraphRecursionError)? Baseline without reranker: 4/15 (2026-10-12 run).

Real LLM calls on the configured provider (~100k tokens on Groq). Pauses between requests keep
under Groq's per-minute limit.

    python -m evals.rerank_ab [--skip-agent]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evals" / "results" / "rerank_ab.json"
PAUSE_SECONDS = 20.0


def rag_refusals(enabled: bool) -> dict:
    from src.rag import RAGPipeline, is_refusal

    os.environ["RERANK_ENABLED"] = "1" if enabled else "0"
    questions = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))["questions"]
    pipeline = RAGPipeline(top_k=4)
    refused = []
    for q in questions:
        time.sleep(PAUSE_SECONDS)
        if is_refusal(pipeline.answer(q["question"], k=4)["answer"]):
            refused.append(q["id"])
    return {"refused": len(refused), "of": len(questions), "ids": refused}


def agent_step_limits() -> dict:
    from src.agent import AgentPipeline

    os.environ["RERANK_ENABLED"] = "1"
    questions = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))["questions"]
    agent = AgentPipeline()
    rows = []
    for q in questions:
        time.sleep(PAUSE_SECONDS)
        result = agent.answer(q["question"])
        rows.append({"id": q["id"], "error": result.get("error"), "tools": [c["tool"] for c in result["tool_calls"]]})
        print(rows[-1], flush=True)
    failed = [r["id"] for r in rows if r["error"] == "GraphRecursionError"]
    other = [r["id"] for r in rows if r["error"] and r["error"] != "GraphRecursionError"]
    return {"step_limit_failures": len(failed), "of": len(rows), "ids": failed, "other_errors": other, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()
    report = {"rag_k4_reranker_off": rag_refusals(False)}
    print(report, flush=True)
    report["rag_k4_reranker_on"] = rag_refusals(True)
    print(report["rag_k4_reranker_on"], flush=True)
    if not args.skip_agent:
        report["agent_reranker_on"] = agent_step_limits()
    report["agent_baseline_without_reranker"] = {"step_limit_failures": 4, "of": 15, "ids": ["q02", "q05", "q07", "q10"],
                                                 "source": "docs/STATUS.md, 2026-10-12 run"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: {x: y for x, y in v.items() if x != "rows"} for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
