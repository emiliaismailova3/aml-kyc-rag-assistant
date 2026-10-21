"""Optional Langfuse tracing. Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set in .env.

What is traced when it is on:
  - every LLMClient call as a "generation" (model, provider, tier, tokens, estimated cost, attempts,
    fallback), so RAG answers, invoice extraction, translation and summaries are covered
  - the agent's full run through Langfuse's LangChain callback (each model step and tool call)
  - retrieval (vector search + rerank) as a "retriever" span, and one span per API request

Every value sent to Langfuse passes through src.security.pii.mask first, so traces never contain
e-mails, phone numbers, IBANs, card numbers or FIN codes.

Settings: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST (e.g. https://cloud.langfuse.com).
Needs `pip install -r requirements.txt` (langfuse). Without keys every helper here is a no-op.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any

from src.security.pii import mask

logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def mask_data(data: Any = None, **kwargs: Any) -> Any:
    """Langfuse mask hook: recursively mask personal data in strings, lists and dicts."""
    if isinstance(data, str):
        return mask(data)
    if isinstance(data, list):
        return [mask_data(data=x) for x in data]
    if isinstance(data, tuple):
        return [mask_data(data=x) for x in data]
    if isinstance(data, dict):
        return {k: mask_data(data=v) for k, v in data.items()}
    return data


@lru_cache(maxsize=1)
def get_langfuse():
    """The shared Langfuse client, or None when tracing is off or the package is missing."""
    if not tracing_enabled():
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("LANGFUSE_* keys are set but the langfuse package is not installed; tracing is off")
        return None
    return Langfuse(mask=mask_data)


def observe(name: str, as_type: str = "span", **fields: Any):
    """Context manager around a Langfuse observation; yields None (and does nothing) when tracing is off.
    Tracing failures are logged and never break the request."""
    client = get_langfuse()
    if client is None:
        return nullcontext(None)
    return _safe_observation(client, name, as_type, fields)


@contextmanager
def _safe_observation(client, name: str, as_type: str, fields: dict):
    try:
        manager = client.start_as_current_observation(name=name, as_type=as_type, **fields)
        observation = manager.__enter__()
    except Exception:  # noqa: BLE001 - tracing must never take a request down
        logger.warning("Could not start a Langfuse observation", exc_info=True)
        yield None
        return
    try:
        yield observation
    except BaseException as exc:
        try:
            observation.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:500])
        except Exception:  # noqa: BLE001
            pass
        manager.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        manager.__exit__(None, None, None)


def update(observation, **fields: Any) -> None:
    if observation is None:
        return
    try:
        observation.update(**fields)
    except Exception:  # noqa: BLE001
        logger.warning("Could not update a Langfuse observation", exc_info=True)


def langchain_callbacks() -> list:
    """Langfuse's LangChain callback handler for the agent, or [] when tracing is off."""
    if get_langfuse() is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:  # noqa: BLE001
        logger.warning("Could not create the Langfuse LangChain handler", exc_info=True)
        return []


def flush() -> None:
    client = get_langfuse()
    if client is not None:
        client.flush()
