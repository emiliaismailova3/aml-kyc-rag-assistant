"""Does the agent still loop? Runs the current agent on the 15 gold questions.

Counts, per question: hit the step limit (the agent kept searching; the base-RAG fallback then
answered), unanswered (an error was returned), refused ("I don't know ..."), and which tools ran.
Baseline before the 3-search cap and the fallback: 4 of 15 ended in the step limit with no answer
(2026-10-12, docs/STATUS.md).

The answers are also written in src/evaluate.py's cache format to
data/eval_results/agent_samples_cache.json, so the agent RAGAS run can score them without
re-asking the agent (the previous cache is kept as agent_samples_cache.before_fixes.json).

    python -m evals.agent_loop_check
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evals" / "results" / "agent_loop_check.json"
CACHE = ROOT / "data" / "eval_results" / "agent_samples_cache.json"
PAUSE_SECONDS = 30.0


def main() -> None:
    from src.agent import AgentPipeline
    from src.rag import is_refusal

    questions = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))["questions"]
    agent = AgentPipeline()
    rows, samples = [], []
    for q in questions:
        time.sleep(PAUSE_SECONDS)
        t = time.perf_counter()
        result = agent.answer(q["question"])
        row = {
            "id": q["id"],
            "step_limit_then_rag": result.get("fallback") == "rag_after_step_limit",
            "error": result.get("error"),
            "refused": is_refusal(result["answer"]),
            "tools": [c["tool"] for c in result["tool_calls"]],
            "kb_searches": sum(c["tool"] == "search_knowledge_base" for c in result["tool_calls"]),
            "seconds": round(time.perf_counter() - t, 1),
        }
        rows.append(row)
        print(row, flush=True)
        if not result.get("error"):
            samples.append({"id": q["id"], "question": q["question"], "answer": result["answer"],
                            "contexts": result.get("contexts") or [""], "ground_truth": q["ground_truth"]})
    n = len(rows)
    report = {
        "questions": n,
        "step_limit_hits": sum(r["step_limit_then_rag"] for r in rows),
        "unanswered_errors": sum(bool(r["error"]) for r in rows),
        "refused": sum(r["refused"] for r in rows),
        "answered": sum(not r["error"] and not r["refused"] for r in rows),
        "baseline_before_fixes": {"step_limit_no_answer": 4, "of": 15, "note": "recursion_limit 10, no tool caps, no fallback"},
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if CACHE.exists():
        shutil.copy(CACHE, CACHE.with_name("agent_samples_cache.before_fixes.json"))
    CACHE.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
