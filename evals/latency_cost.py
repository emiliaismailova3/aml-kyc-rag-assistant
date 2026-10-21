"""Latency, tokens and estimated cost per request type, plus end-to-end invoice -> company matching.

Request types (all real LLM calls on the configured provider, Groq free tier by default):
  rag      - RAGPipeline.answer on the first N gold questions
  sql      - the agent on the SQL routing scenarios (t11-t13); needs `python -m scripts.seed_demo_db`
  invoice  - extract_invoice on all 15 test invoices, then CompanyMatcher (with and without VOEN)

Latency is wall-clock per request. Tokens and cost come from the llm_calls table rows written during
this run (the same logging the app uses). Cost is ESTIMATED from the list prices in src/config.py;
on Groq's free tier nothing is actually billed.

Routing saving: for calls the router sent to the small model, the same token counts are re-priced at
the main model's price. This is a counterfactual price comparison; it does not measure whether the
main model would have given better or worse output.

    python -m evals.latency_cost [--rag-questions 10]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from sqlalchemy import select

from src.llm_client import LLMUnavailable

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evals" / "results" / "latency_cost.json"
# Groq's free tier allows 8000 tokens per minute, about 3 RAG answers. A pause between requests
# keeps the run under that limit; it is NOT included in the measured latency.
PAUSE_SECONDS = 20.0
FAILED: dict[str, int] = {}


def timed(kind: str, fn) -> float | None:
    """Run one request; return its latency in ms, or None (counted in FAILED) if every provider failed."""
    time.sleep(PAUSE_SECONDS)
    t = time.perf_counter()
    try:
        fn()
    except LLMUnavailable:
        FAILED[kind] = FAILED.get(kind, 0) + 1
        return None
    return (time.perf_counter() - t) * 1000


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (rank - low), 1)


def summary(latencies_ms: list[float]) -> dict:
    return {"n": len(latencies_ms), "p50_ms": pct(latencies_ms, 0.5), "p95_ms": pct(latencies_ms, 0.95),
            "mean_ms": round(statistics.mean(latencies_ms), 1)}


def calls_since(started_at, tasks: set[str]) -> list[dict]:
    from src.db import get_engine, llm_calls

    with get_engine().connect() as conn:
        rows = conn.execute(select(llm_calls).where(llm_calls.c.timestamp >= started_at)).mappings().all()
    return [dict(r) for r in rows if r["task"] in tasks]


def token_cost(rows: list[dict], requests: int) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    tin, tout = sum(r["input_tokens"] for r in ok), sum(r["output_tokens"] for r in ok)
    cost = sum(r["cost_usd"] for r in ok)
    return {"llm_calls": len(rows), "failed_calls": len(rows) - len(ok),
            "avg_input_tokens_per_request": round(tin / requests, 1), "avg_output_tokens_per_request": round(tout / requests, 1),
            "avg_est_cost_usd_per_request": round(cost / requests, 6), "total_est_cost_usd": round(cost, 6)}


def run_rag(n: int) -> dict:
    from src.rag import RAGPipeline

    questions = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))["questions"][:n]
    pipeline = RAGPipeline()
    latencies = [timed("rag", lambda q=q: pipeline.answer(q["question"])) for q in questions]
    return {"latencies": [x for x in latencies if x is not None]}


def run_sql() -> dict:
    from src.agent import AgentPipeline

    scenarios = json.loads((ROOT / "data" / "agent_test_scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    agent = AgentPipeline()
    latencies, used = [], []
    for s in [s for s in scenarios if s["expected_tool"] == "sql_query"]:
        time.sleep(PAUSE_SECONDS)
        t = time.perf_counter()
        result = agent.answer(s["question"])
        latencies.append((time.perf_counter() - t) * 1000)
        used.append({"id": s["id"], "tools": [c["tool"] for c in result["tool_calls"]], "error": result.get("error")})
    return {"latencies": latencies, "runs": used}


def run_invoices() -> dict:
    from src.invoices.evaluate import field_matches
    from src.invoices.extract import extract_invoice
    from src.matching.matcher import CompanyMatcher, load_companies

    labels = json.loads((ROOT / "data" / "invoices" / "labels.json").read_text(encoding="utf-8"))
    matcher = CompanyMatcher(load_companies())
    latencies, rows = [], []
    for label in labels:
        time.sleep(PAUSE_SECONDS / 2)
        t = time.perf_counter()
        result = extract_invoice(ROOT / "data" / "invoices" / label["file"])
        latencies.append((time.perf_counter() - t) * 1000)
        inv = result.invoice
        row = {"file": label["file"], "kind": label["kind"], "attempts": result.attempts,
               "needs_human_review": result.needs_human_review}
        if inv is not None:
            row["fields_correct"] = {f: field_matches(f, getattr(inv, f), label[f])
                                     for f in ("company_name", "voen", "invoice_number", "issue_date", "total_amount", "currency")}
            with_voen = matcher.match(inv.company_name, voen=inv.voen)
            name_only = matcher.match(inv.company_name)
            row["match_with_voen"] = {"correct": with_voen.company_id == label["reference_company_id"] and not with_voen.needs_human_review,
                                      "method": with_voen.method}
            row["match_name_only"] = {"correct": name_only.company_id == label["reference_company_id"] and not name_only.needs_human_review,
                                      "review": name_only.needs_human_review, "method": name_only.method}
        rows.append(row)
    return {"latencies": latencies, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-questions", type=int, default=10)
    parser.add_argument("--only", choices=["rag", "sql_via_agent", "invoice_extraction"], help="run one request type")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from src.config import get_prices
    from src.db import utcnow

    report: dict = {}
    for name, runner, tasks in (
        ("rag", lambda: run_rag(args.rag_questions), {"rag_answer"}),
        ("sql_via_agent", run_sql, {"agent"}),
        ("invoice_extraction", run_invoices, {"invoice_extraction"}),
    ):
        if args.only and name != args.only:
            continue
        started_at = utcnow()
        out = runner()
        calls = calls_since(started_at, tasks)
        report[name] = {"latency": summary(out["latencies"]), "failed_requests": FAILED.get(name.split("_")[0], 0), **token_cost(calls, len(out["latencies"])),
                        **{k: v for k, v in out.items() if k != "latencies"},
                        "tiers": sorted({c["tier"] for c in calls}), "models": sorted({c["model"] for c in calls})}
        report[name]["_calls"] = calls
        print(name, report[name]["latency"], report[name]["avg_est_cost_usd_per_request"])

    # Invoice accuracy and matching on this run
    if "invoice_extraction" not in report:
        report["invoice_extraction"] = {"rows": [], "_calls": []}
    inv_rows = [r for r in report["invoice_extraction"]["rows"] if "fields_correct" in r]
    n = len(report["invoice_extraction"]["rows"]) or 1
    report["invoice_extraction"]["accuracy_this_run"] = {
        "extracted": len(inv_rows), "of": n,
        **{f: round(sum(r["fields_correct"][f] for r in inv_rows) / n, 3)
           for f in ("company_name", "voen", "invoice_number", "issue_date", "total_amount", "currency")},
        "company_matched_correctly_with_voen": round(sum(r["match_with_voen"]["correct"] for r in inv_rows) / n, 3),
        "company_matched_correctly_name_only": round(sum(r["match_name_only"]["correct"] for r in inv_rows) / n, 3),
        "name_only_sent_to_review": sum(r["match_name_only"]["review"] for r in inv_rows),
    }

    # Small-model routing: re-price small-tier calls at the main model's price.
    prices = get_prices()
    small_calls = [c for part in report.values() for c in part["_calls"] if c["tier"] == "small" and c["status"] == "ok"]
    if small_calls:
        main_model = "openai/gpt-oss-120b"
        pin, pout = prices[main_model]
        actual = sum(c["cost_usd"] for c in small_calls)
        if_main = sum((c["input_tokens"] * pin + c["output_tokens"] * pout) / 1e6 for c in small_calls)
        report["small_model_routing"] = {
            "small_tier_calls": len(small_calls), "est_cost_small_usd": round(actual, 6),
            "est_cost_if_main_usd": round(if_main, 6), "est_saving_pct": round(100 * (1 - actual / if_main), 1),
            "note": "same token counts re-priced; output quality of the main model on these calls not measured",
        }
    for part in report.values():
        part.pop("_calls", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: {x: y for x, y in v.items() if x not in ("rows", "runs")} for k, v in report.items()},
                     indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
