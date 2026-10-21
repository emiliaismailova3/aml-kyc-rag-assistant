"""Forced-failure test of the LLM reliability layer, against a REAL provider.

Scenario A: the primary provider is "down" (it points at a closed local port, so every call fails
with a connection error) and the real configured provider (Groq) is second in the chain.
Expected: retries with backoff on the dead one, then fallback, and a real answer.

Scenario B: every provider is down. Expected: LLMUnavailable with needs_human=True.

Nothing is sent anywhere except the one real call in scenario A. Uses ~50 tokens.

    python -m evals.forced_failover
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from src.config import get_provider_chain
from src.llm_client import LLMClient, LLMUnavailable, build_provider

OUT = Path(__file__).resolve().parent / "results" / "forced_failover.json"
DEAD_URL = "http://127.0.0.1:9/v1"  # nothing listens on port 9 ("discard")
MESSAGES = [{"role": "user", "content": "Reply with the single word: ok"}]


def run(providers) -> dict:
    log: list[dict] = []
    client = LLMClient(providers, recorder=log.append)
    started = time.perf_counter()
    try:
        reply = client.complete(MESSAGES, task="forced_failover")
        outcome = {"answered": True, "text": reply.text.strip()[:40], "provider": reply.provider,
                   "fell_back": reply.fell_back, "attempts": reply.attempts, "errors": reply.errors}
    except LLMUnavailable as exc:
        outcome = {"answered": False, "needs_human": exc.needs_human, "rate_limited": exc.rate_limited,
                   "message": str(exc)[:200]}
    outcome["seconds"] = round(time.perf_counter() - started, 1)
    outcome["logged_attempts"] = [{"provider": r["provider"], "status": r["status"], "error": r["error"]} for r in log]
    return outcome


def main() -> None:
    real = get_provider_chain()
    if not real:
        raise SystemExit("No provider with an API key configured in .env")
    dead = replace(real[0], name="dead-primary", base_url=DEAD_URL)
    report = {
        "A_primary_down_then_real_provider": run([build_provider(dead), build_provider(real[0])]),
        "B_all_providers_down": run([build_provider(dead), build_provider(replace(dead, name="dead-secondary"))]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
