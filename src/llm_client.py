"""One wrapper for LLM calls: routing, retries, fallback providers, logging.

For every call:
  1. route      - pick the small or the main model (see choose_tier)
  2. try        - call the first provider; on a transient failure (429, 5xx, timeout,
                  dropped connection) wait with exponential backoff + jitter and retry
  3. fall back  - if a provider keeps failing, move to the next one in LLM_FALLBACK_CHAIN
  4. escalate   - if every provider failed, raise LLMUnavailable(needs_human=True)
  5. log        - every attempt is recorded: model, latency, tokens, estimated cost,
                  status, error type. Prompts and answers are never stored.

Usage:
    from src.llm_client import get_client
    reply = get_client().complete([{"role": "user", "content": "Hi"}], task="demo")
    print(reply.text, reply.cost_usd)
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

from src.config import ProviderSettings, get_prices, get_provider_chain

logger = logging.getLogger(__name__)

# Up to 1 + MAX_RETRIES attempts per provider before falling back to the next one.
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0
REQUEST_TIMEOUT_SECONDS = 60.0

# Routing rule (explainable on purpose): short, tool-free prompts go to the small
# model. About 4 characters per token, so 4000 characters is roughly 1000 tokens.
SMALL_MAX_CHARS = 4000

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}


class LLMUnavailable(Exception):
    """Every provider failed. The caller should hand the request to a human."""

    needs_human = True

    def __init__(self, message: str, rate_limited: bool = False):
        super().__init__(message)
        self.rate_limited = rate_limited


@dataclass
class RawReply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    tier: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    attempts: int
    fell_back: bool = False
    errors: list[str] = field(default_factory=list)


# --- routing ------------------------------------------------------------------

def choose_tier(messages: list[Message], needs_tools: bool = False, retry: bool = False) -> str:
    """"small" for a short plain prompt; "main" when tools are needed, when this is a
    retry after the small model already failed, or when the prompt is long."""
    if needs_tools or retry:
        return "main"
    total_chars = sum(len(m["content"]) for m in messages)
    return "small" if total_chars <= SMALL_MAX_CHARS else "main"


# --- retry helpers -----------------------------------------------------------------

def is_retryable(exc: Exception) -> bool:
    """Transient problems only: rate limit, request timeout, server errors, network drops.
    A 400/401/404 will not fix itself, so retrying it only wastes time."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "TimeoutError"}


def backoff_delay(attempt: int, rng: random.Random | None = None) -> float:
    """Exponential backoff with jitter: 1s, 2s, 4s... (capped), randomly shortened to 50-100%
    so many clients that failed together do not all retry at the same instant."""
    rng = rng or random
    ceiling = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2**attempt)
    return rng.uniform(ceiling / 2, ceiling)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = get_prices().get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


# --- providers ---------------------------------------------------------------------

class Provider:
    """Thin adapter: (messages, model) -> RawReply. Subclasses wrap one vendor SDK."""

    def __init__(self, settings: ProviderSettings):
        self.settings = settings
        self.name = settings.name

    def model_for(self, tier: str) -> str:
        return self.settings.small_model if tier == "small" and self.settings.small_model else self.settings.main_model

    def call(self, messages: list[Message], model: str, max_tokens: int | None) -> RawReply:
        raise NotImplementedError


class OpenAICompatibleProvider(Provider):
    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        from openai import OpenAI

        # max_retries=0: retries are done here, where they can be logged and can trigger fallback.
        self._client = OpenAI(
            api_key=settings.api_key, base_url=settings.base_url, max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS
        )

    def call(self, messages, model, max_tokens):
        response = self._client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=max_tokens
        )
        usage = response.usage
        return RawReply(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


class AnthropicProvider(Provider):
    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        import anthropic

        self._client = anthropic.Anthropic(
            api_key=settings.api_key, max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS
        )

    def call(self, messages, model, max_tokens):
        # Anthropic takes the system prompt as a separate argument, not as a message.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        options = {"model": model, "messages": chat, "temperature": 0, "max_tokens": max_tokens or 4096}
        if system:
            options["system"] = system
        response = self._client.messages.create(**options)
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        return RawReply(text=text, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)


def build_provider(settings: ProviderSettings) -> Provider:
    return AnthropicProvider(settings) if settings.kind == "anthropic" else OpenAICompatibleProvider(settings)


# --- logging -----------------------------------------------------------------------

def db_recorder(row: dict) -> None:
    """Default call logger: one row in the llm_calls table (SQLite or PostgreSQL)."""
    from sqlalchemy import insert

    from src.db import get_engine, init_db, llm_calls, utcnow

    engine = get_engine()
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(insert(llm_calls).values(timestamp=utcnow(), **row))


# --- the client --------------------------------------------------------------------

class LLMClient:
    def __init__(
        self,
        providers: list[Provider],
        recorder: Callable[[dict], None] | None = db_recorder,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ):
        self.providers = providers
        self.recorder = recorder
        self._sleep = sleep
        self._rng = rng

    def _record(self, **row) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder(row)
        except Exception:  # noqa: BLE001 - logging must never break a user request
            logger.warning("Could not record LLM call", exc_info=True)

    def complete(
        self,
        messages: list[Message],
        task: str = "chat",
        tier: str | None = None,
        needs_tools: bool = False,
        retry: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not self.providers:
            raise LLMUnavailable("No LLM provider is configured: set an API key in .env.")
        tier = tier or choose_tier(messages, needs_tools=needs_tools, retry=retry)

        errors: list[str] = []
        rate_limited = False
        attempts = 0
        for position, provider in enumerate(self.providers):
            model = provider.model_for(tier)
            for retry_number in range(MAX_RETRIES + 1):
                attempts += 1
                started = time.perf_counter()
                try:
                    raw = provider.call(messages, model, max_tokens)
                except Exception as exc:  # noqa: BLE001 - any SDK error is classified below
                    latency_ms = (time.perf_counter() - started) * 1000
                    error_name = type(exc).__name__
                    errors.append(f"{provider.name}/{model}: {error_name}")
                    rate_limited = getattr(exc, "status_code", None) == 429
                    self._record(
                        provider=provider.name, model=model, task=task, tier=tier, latency_ms=latency_ms,
                        input_tokens=0, output_tokens=0, cost_usd=0.0, status="error", error=error_name,
                    )
                    if not is_retryable(exc) or retry_number == MAX_RETRIES:
                        logger.warning("Provider %s gave up after %d attempt(s): %s", provider.name, retry_number + 1, exc)
                        break  # next provider
                    delay = backoff_delay(retry_number, self._rng)
                    logger.info("Provider %s failed (%s); retrying in %.1fs", provider.name, error_name, delay)
                    self._sleep(delay)
                    continue

                latency_ms = (time.perf_counter() - started) * 1000
                cost = estimate_cost(model, raw.input_tokens, raw.output_tokens)
                self._record(
                    provider=provider.name, model=model, task=task, tier=tier, latency_ms=latency_ms,
                    input_tokens=raw.input_tokens, output_tokens=raw.output_tokens, cost_usd=cost,
                    status="ok", error=None,
                )
                return LLMResponse(
                    text=raw.text, provider=provider.name, model=model, tier=tier,
                    input_tokens=raw.input_tokens, output_tokens=raw.output_tokens, cost_usd=cost,
                    latency_ms=latency_ms, attempts=attempts, fell_back=position > 0, errors=errors,
                )

        raise LLMUnavailable(
            "All LLM providers failed (" + "; ".join(errors[-4:]) + "). Escalating to a human.",
            rate_limited=rate_limited,
        )


@lru_cache(maxsize=1)
def get_client() -> LLMClient:
    return LLMClient([build_provider(settings) for settings in get_provider_chain()])
