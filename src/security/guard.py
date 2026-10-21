"""The two checkpoints the API applies to every chat request.

  screen_question(q)  -> refuse prompt-injection attempts; mask PII before the LLM sees the question
  check_answer(a)     -> withhold an answer that contains the system-prompt canary

PII_MASK_PROMPTS=0 in .env turns off masking of what is sent to the LLM (logs are always masked).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from src.security.injection import REFUSAL, detect_injection, leaks_canary
from src.security.pii import mask_pii

logger = logging.getLogger(__name__)


class InjectionBlocked(Exception):
    def __init__(self, rules: list[str]):
        super().__init__("The request looks like a prompt-injection attempt and was refused.")
        self.rules = rules


def mask_prompts_enabled() -> bool:
    return os.getenv("PII_MASK_PROMPTS", "1").strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class ScreenedQuestion:
    text: str            # what is sent to the LLM
    masked: dict         # entity -> count that was masked


def screen_question(question: str) -> ScreenedQuestion:
    check = detect_injection(question)
    if check.flagged:
        logger.warning("Blocked a prompt-injection attempt (rules: %s)", ", ".join(check.rules))
        raise InjectionBlocked(check.rules)
    if not mask_prompts_enabled():
        return ScreenedQuestion(question, {})
    masked, found = mask_pii(question)
    return ScreenedQuestion(masked, dict(found))


def check_answer(answer: str) -> tuple[str, bool]:
    """(answer to return, leaked?)"""
    if leaks_canary(answer):
        logger.error("An answer contained the system-prompt canary; withholding it")
        return REFUSAL, True
    return answer, False
