"""LangChain callback that logs the agent's LLM calls to the same llm_calls table.

The agent (LangGraph) talks to the model through LangChain's ChatOpenAI, not
through src/llm_client.py, so its calls are recorded by this callback instead:
latency, token counts, estimated cost, status. Retries inside the agent are still
the OpenAI client's built-in ones, and there is no provider fallback for the agent
(see docs/STATUS.md).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from src.llm_client import db_recorder, estimate_cost

logger = logging.getLogger(__name__)


class LLMCallLogger(BaseCallbackHandler):
    def __init__(self, provider: str, model: str, task: str = "agent", recorder=db_recorder):
        self.provider, self.model, self.task, self.recorder = provider, model, task, recorder
        self._started: dict[Any, float] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._started[run_id] = time.perf_counter()

    def _latency_ms(self, run_id) -> float:
        return (time.perf_counter() - self._started.pop(run_id, time.perf_counter())) * 1000

    def _record(self, **row) -> None:
        try:
            self.recorder({"provider": self.provider, "model": self.model, "task": self.task, "tier": "main", **row})
        except Exception:  # noqa: BLE001 - logging must never break the agent
            logger.warning("Could not record agent LLM call", exc_info=True)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        usage = {}
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (AttributeError, IndexError):
            pass
        input_tokens, output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        self._record(
            latency_ms=self._latency_ms(run_id), input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=estimate_cost(self.model, input_tokens, output_tokens), status="ok", error=None,
        )

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._record(
            latency_ms=self._latency_ms(run_id), input_tokens=0, output_tokens=0, cost_usd=0.0,
            status="error", error=type(error).__name__,
        )
