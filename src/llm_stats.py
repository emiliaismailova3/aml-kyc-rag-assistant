"""Aggregate statistics over the llm_calls table (served by GET /stats)."""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.engine import Engine

from src.db import get_engine, init_db, llm_calls

# Only the most recent rows are read, so /stats stays fast on a long-running service.
MAX_ROWS = 20_000


def get_stats(engine: Engine | None = None) -> dict:
    engine = engine or get_engine()
    init_db(engine)
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                llm_calls.c.model, llm_calls.c.latency_ms, llm_calls.c.input_tokens, llm_calls.c.output_tokens,
                llm_calls.c.cost_usd, llm_calls.c.status,
            ).order_by(llm_calls.c.id.desc()).limit(MAX_ROWS)
        ).all()

    if not rows:
        return {"calls": 0, "error_rate": None, "latency_ms": {"p50": None, "p95": None}, "total_cost_usd": 0.0,
                "input_tokens": 0, "output_tokens": 0, "by_model": {}}

    ok_latencies = [r.latency_ms for r in rows if r.status == "ok"]
    errors = sum(r.status != "ok" for r in rows)
    by_model: dict[str, dict] = {}
    for r in rows:
        entry = by_model.setdefault(r.model, {"calls": 0, "errors": 0, "cost_usd": 0.0})
        entry["calls"] += 1
        entry["errors"] += r.status != "ok"
        entry["cost_usd"] = round(entry["cost_usd"] + r.cost_usd, 6)
    return {
        "calls": len(rows),
        "error_rate": round(errors / len(rows), 4),
        # Percentiles describe successful calls only: a failed call's latency is mostly the timeout.
        "latency_ms": {
            "p50": round(float(np.percentile(ok_latencies, 50)), 1) if ok_latencies else None,
            "p95": round(float(np.percentile(ok_latencies, 95)), 1) if ok_latencies else None,
        },
        "total_cost_usd": round(sum(r.cost_usd for r in rows), 6),
        "input_tokens": sum(r.input_tokens for r in rows),
        "output_tokens": sum(r.output_tokens for r in rows),
        "by_model": by_model,
    }
