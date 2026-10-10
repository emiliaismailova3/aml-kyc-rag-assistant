"""Run the agent on the routing scenarios and check which tools it actually called.

PASS rules:
  - tool scenarios: the expected tool was called at least once
  - rag scenarios:  none of calculator / internet_search / sql_query / extract_invoice was called

This costs real LLM calls (one agent run per scenario) and needs the demo database
(`python -m scripts.seed_demo_db`) for the SQL scenarios.

Usage:
    python -m scripts.check_agent_routing            # all scenarios
    python -m scripts.check_agent_routing t11 t14    # only these ids
"""

from __future__ import annotations

import json
import sys
import time

from src.agent import AgentPipeline
from src.config import PROJECT_ROOT

SCENARIOS_PATH = PROJECT_ROOT / "data" / "agent_test_scenarios.json"
RESULT_PATH = PROJECT_ROOT / "data" / "eval_results" / "agent_routing.json"
NON_KNOWLEDGE_TOOLS = {"calculator", "internet_search", "sql_query", "extract_invoice"}


def passed(scenario: dict, tools_used: set[str]) -> bool:
    if scenario["expected_route"] == "tool":
        return scenario["expected_tool"] in tools_used
    return not (tools_used & NON_KNOWLEDGE_TOOLS)


def main(ids: list[str]) -> None:
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))["scenarios"]
    if ids:
        scenarios = [s for s in scenarios if s["id"] in ids]
    agent = AgentPipeline()
    results = []
    for scenario in scenarios:
        result = agent.answer(scenario["question"])
        tools_used = {call["tool"] for call in result["tool_calls"]}
        ok = passed(scenario, tools_used) and not result.get("error")
        results.append(
            {"id": scenario["id"], "expected_tool": scenario["expected_tool"], "tools_used": sorted(tools_used),
             "passed": ok, "error": result.get("error"), "answer": result["answer"][:300]}
        )
        print(f"{scenario['id']}: {'PASS' if ok else 'FAIL'}  expected={scenario['expected_tool']}  used={sorted(tools_used)}")
        time.sleep(2)  # be gentle with the provider's rate limit
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{sum(r['passed'] for r in results)}/{len(results)} passed; saved to {RESULT_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:])
