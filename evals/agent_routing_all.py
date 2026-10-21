"""Tool-selection accuracy of the agent on all 16 scenarios in data/agent_test_scenarios.json.

Same pass rule as scripts/check_agent_routing.py (expected tool called; for knowledge-base questions,
no calculator / web / SQL / invoice tool called), but runs every scenario, pauses between them for
Groq's per-minute token limit, and writes to evals/results/ so the committed
data/eval_results/agent_routing.json is not overwritten.

    python -m evals.agent_routing_all
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from scripts.check_agent_routing import SCENARIOS_PATH, passed
from src.agent import AgentPipeline

OUT = Path(__file__).resolve().parent / "results" / "agent_routing_all.json"
PAUSE_SECONDS = 20.0


def main() -> None:
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))["scenarios"]
    agent = AgentPipeline()
    rows = []
    for s in scenarios:
        time.sleep(PAUSE_SECONDS)
        t = time.perf_counter()
        result = agent.answer(s["question"])
        tools = sorted({c["tool"] for c in result["tool_calls"]})
        ok = passed(s, set(tools)) and not result.get("error")
        rows.append({"id": s["id"], "expected": s["expected_tool"] or "search_knowledge_base (rag)", "used": tools,
                     "passed": ok, "error": result.get("error"), "seconds": round(time.perf_counter() - t, 1)})
        print(rows[-1])
    by_expected = Counter(r["expected"] for r in rows)
    report = {
        "scenarios": len(rows), "passed": sum(r["passed"] for r in rows),
        "accuracy": round(sum(r["passed"] for r in rows) / len(rows), 3),
        "by_expected_tool": {k: f"{sum(r['passed'] for r in rows if r['expected'] == k)}/{v}" for k, v in by_expected.items()},
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
