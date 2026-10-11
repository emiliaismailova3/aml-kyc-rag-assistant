"""Polite downloading: respect robots.txt and never hit one site faster than a minimum interval."""

from __future__ import annotations

import logging
import threading
import time
from urllib import robotparser
from urllib.parse import urlparse

import requests

from src.config import COLLECT_MIN_INTERVAL_SECONDS, COLLECTOR_USER_AGENT

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 20
MAX_BYTES = 15 * 1024 * 1024  # refuse anything bigger than 15 MB


class DisallowedByRobots(Exception):
    """robots.txt forbids fetching this URL. Retrying will not help."""


class UnsupportedContent(Exception):
    """The response is neither HTML nor PDF (or is too large). Retrying will not help."""


class RateLimiter:
    """Allows one request per `min_interval` seconds for each host (thread-safe)."""

    def __init__(self, min_interval: float = COLLECT_MIN_INTERVAL_SECONDS, clock=time.monotonic, sleep=time.sleep):
        self.min_interval = min_interval
        self._clock, self._sleep = clock, sleep
        self._next_allowed: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> float:
        """Block until `host` may be requested again; return how long we waited."""
        with self._lock:
            now = self._clock()
            ready_at = self._next_allowed.get(host, now)
            wait = max(0.0, ready_at - now)
            self._next_allowed[host] = max(now, ready_at) + self.min_interval
        if wait:
            self._sleep(wait)
        return wait


_robots_cache: dict[str, robotparser.RobotFileParser] = {}
_limiter = RateLimiter()


def _robots_for(origin: str, session: requests.Session) -> robotparser.RobotFileParser:
    if origin not in _robots_cache:
        parser = robotparser.RobotFileParser()
        try:
            response = session.get(f"{origin}/robots.txt", timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            elif response.status_code in (401, 403):
                parser.disallow_all = True  # the site refuses us: treat as "do not crawl"
            else:
                parser.parse([])  # no robots.txt (404 etc.): everything is allowed
        except requests.RequestException:
            parser.parse([])
        _robots_cache[origin] = parser
    return _robots_cache[origin]


def is_allowed(url: str, session: requests.Session | None = None) -> bool:
    session = session or requests.Session()
    session.headers["User-Agent"] = COLLECTOR_USER_AGENT
    parts = urlparse(url)
    return _robots_for(f"{parts.scheme}://{parts.netloc}", session).can_fetch(COLLECTOR_USER_AGENT, url)


def fetch(url: str, limiter: RateLimiter | None = None) -> tuple[bytes, str]:
    """Download `url`; returns (body, content_type). Raises DisallowedByRobots / UnsupportedContent
    (no point retrying) or requests exceptions (worth retrying)."""
    limiter = limiter or _limiter
    session = requests.Session()
    session.headers["User-Agent"] = COLLECTOR_USER_AGENT
    if not is_allowed(url, session):
        raise DisallowedByRobots(url)
    limiter.wait(urlparse(url).netloc)
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if content_type not in {"text/html", "application/xhtml+xml", "application/pdf"}:
        raise UnsupportedContent(f"{url} has content type {content_type!r}")
    if len(response.content) > MAX_BYTES:
        raise UnsupportedContent(f"{url} is larger than {MAX_BYTES} bytes")
    return response.content, content_type
