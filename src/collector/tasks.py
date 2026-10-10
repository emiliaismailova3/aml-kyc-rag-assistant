"""Celery tasks: fetch configured sources on a schedule (Celery beat + Redis).

Run (Redis must be up, e.g. `docker compose --profile full up -d redis`):
    celery -A src.collector.tasks worker --loglevel=info          # Linux/macOS/Docker
    celery -A src.collector.tasks worker --pool=solo --loglevel=info   # Windows
    celery -A src.collector.tasks beat --loglevel=info            # the scheduler

Trigger once by hand:
    python -c "from src.collector.tasks import collect_all; collect_all.delay()"
"""

from __future__ import annotations

import json
import logging

import requests
from celery import Celery

from src.collector.fetch import DisallowedByRobots, UnsupportedContent
from src.collector.pipeline import collect_url
from src.config import COLLECT_INTERVAL_MINUTES, PROJECT_ROOT, REDIS_URL

logger = logging.getLogger(__name__)

SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"

app = Celery("collector", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_acks_late=True,                 # a task is acknowledged only after it finished: a crashed worker's task is redelivered
    worker_prefetch_multiplier=1,        # fetch one task at a time (politeness)
    broker_connection_retry_on_startup=True,
    result_expires=3600,
    beat_schedule={
        "collect-all-sources": {
            "task": "src.collector.tasks.collect_all",
            "schedule": COLLECT_INTERVAL_MINUTES * 60.0,
        }
    },
)


def load_sources() -> list[dict]:
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))["sources"]


@app.task(
    bind=True,
    name="src.collector.tasks.collect_one",
    autoretry_for=(requests.RequestException,),   # network problems, 5xx: worth retrying
    retry_backoff=30,                              # 30s, 60s, 120s ...
    retry_backoff_max=600,
    retry_jitter=True,
    max_retries=3,
    rate_limit="30/m",
)
def collect_one(self, url: str) -> dict:
    try:
        return collect_url(url)
    except (DisallowedByRobots, UnsupportedContent) as exc:
        # Permanent problems: report them, do not retry.
        logger.warning("Skipping %s: %s", url, type(exc).__name__)
        return {"url": url, "status": "skipped", "reason": type(exc).__name__}


@app.task(name="src.collector.tasks.collect_all")
def collect_all() -> int:
    """Queue one task per configured source (so each can retry independently)."""
    sources = load_sources()
    for source in sources:
        collect_one.delay(source["url"])
    return len(sources)
