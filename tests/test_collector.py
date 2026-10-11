"""Tests for the scheduled data collector (src/collector/).

Network is replaced by fakes. Tests marked `redis` use a real Redis broker and a
real in-process Celery worker; they are skipped when Redis is not reachable
(`docker compose --profile full up -d redis`).
"""

import json
from pathlib import Path

import pytest
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine

from src.collector import fetch as fetch_module
from src.collector import tasks
from src.collector.clean import content_hash, extract_text, html_to_text, normalize_text, pdf_to_text
from src.collector.fetch import DisallowedByRobots, RateLimiter, UnsupportedContent, fetch, is_allowed
from src.collector.pipeline import add_chunks, collect_url, doc_name_for, remove_document
from src.config import REDIS_URL
from src.db import collected_documents, count_rows

ROOT = Path(__file__).resolve().parent.parent
LONG_TEXT = "Customer due diligence means identifying and verifying the customer. " * 12


# --- cleaning ---------------------------------------------------------------------

def test_html_to_text_drops_scripts_navigation_and_footer():
    html = """<html><body><nav>Menu Home</nav><script>alert(1)</script>
    <main><h1>Title</h1><p>Real   content here.</p></main><footer>Copyright</footer></body></html>"""
    text = normalize_text(html_to_text(html))
    assert "Real content here." in text and "Title" in text
    assert "Menu" not in text and "alert" not in text and "Copyright" not in text


def test_normalize_text_unifies_whitespace_and_unicode():
    assert normalize_text("a  b \t c\n\n\n\n d­e") == "a b c\n\nde"


def test_content_hash_ignores_case_and_whitespace_but_not_words():
    assert content_hash("Hello   World\n") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("hello there")


def test_pdf_text_extraction():
    data = (ROOT / "data" / "invoices" / "inv_01_text.pdf").read_bytes()
    assert "VOEN" in pdf_to_text(data)
    assert "VOEN" in extract_text(data, "application/pdf")


# --- politeness ---------------------------------------------------------------------

def test_rate_limiter_spaces_out_requests_to_the_same_host_only():
    now = [0.0]
    slept = []
    limiter = RateLimiter(min_interval=5, clock=lambda: now[0], sleep=slept.append)
    assert limiter.wait("a.com") == 0          # first request: no wait
    assert limiter.wait("a.com") == 5          # second one right after: wait the full interval
    assert limiter.wait("b.com") == 0          # a different site is independent
    now[0] = 100.0
    assert limiter.wait("a.com") == 0          # long enough ago: no wait
    assert slept == [5]


class FakeResponse:
    def __init__(self, status=200, text="", content=b"", content_type="text/html"):
        self.status_code, self.text, self.content = status, text, content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.requested = routes, {}, []

    def get(self, url, timeout=None):
        self.requested.append(url)
        return self.routes.get(url, FakeResponse(404))


@pytest.fixture(autouse=True)
def clear_robots_cache():
    fetch_module._robots_cache.clear()
    yield
    fetch_module._robots_cache.clear()


def _patch_session(monkeypatch, routes):
    session = FakeSession(routes)
    monkeypatch.setattr(fetch_module.requests, "Session", lambda: session)
    return session


def test_robots_txt_disallow_is_respected(monkeypatch):
    routes = {"https://site.test/robots.txt": FakeResponse(text="User-agent: *\nDisallow: /private/")}
    session = FakeSession(routes)
    assert is_allowed("https://site.test/public/page", session) is True
    assert is_allowed("https://site.test/private/page", session) is False


def test_missing_robots_txt_means_allowed_and_forbidden_robots_means_blocked():
    assert is_allowed("https://open.test/x", FakeSession({})) is True
    forbidden = FakeSession({"https://closed.test/robots.txt": FakeResponse(403)})
    assert is_allowed("https://closed.test/x", forbidden) is False


def test_fetch_refuses_disallowed_urls_without_downloading(monkeypatch):
    session = _patch_session(monkeypatch, {"https://site.test/robots.txt": FakeResponse(text="User-agent: *\nDisallow: /")})
    with pytest.raises(DisallowedByRobots):
        fetch("https://site.test/page", limiter=RateLimiter(0))
    assert "https://site.test/page" not in session.requested


def test_fetch_returns_body_and_content_type(monkeypatch):
    _patch_session(monkeypatch, {"https://site.test/page": FakeResponse(content=b"<p>hi</p>", content_type="text/html")})
    assert fetch("https://site.test/page", limiter=RateLimiter(0)) == (b"<p>hi</p>", "text/html")


def test_fetch_rejects_other_content_types(monkeypatch):
    _patch_session(monkeypatch, {"https://site.test/img": FakeResponse(content=b"x", content_type="image/png")})
    with pytest.raises(UnsupportedContent):
        fetch("https://site.test/img", limiter=RateLimiter(0))


def test_fetch_raises_http_errors_so_celery_can_retry(monkeypatch):
    _patch_session(monkeypatch, {"https://site.test/down": FakeResponse(503)})
    with pytest.raises(requests.HTTPError):
        fetch("https://site.test/down", limiter=RateLimiter(0))


# --- pipeline -------------------------------------------------------------------------

class RecordingStore:
    def __init__(self):
        self.added, self.removed = [], []

    def add_documents(self, chunks, ids=None):
        self.added.append((len(chunks), ids))

    def _collection_delete(self, where):
        self.removed.append(where)

    @property
    def _collection(self):
        outer = self

        class Collection:
            def delete(self, where):
                outer._collection_delete(where)

        return Collection()


@pytest.fixture
def engine(tmp_path):
    return create_engine(f"sqlite:///{(tmp_path / 'collector.db').as_posix()}")


def page(text):
    return lambda url: (f"<html><body><main>{text}</main></body></html>".encode(), "text/html")


def test_new_page_is_ingested_and_recorded(engine):
    store = RecordingStore()
    result = collect_url("https://site.test/kyc", fetcher=page(LONG_TEXT), store=store, engine=engine)
    assert result["status"] == "ingested" and result["chunks"] >= 1
    assert store.added and count_rows(collected_documents, engine) == 1


def test_same_content_is_not_ingested_twice(engine):
    store = RecordingStore()
    collect_url("https://site.test/kyc", fetcher=page(LONG_TEXT), store=store, engine=engine)
    again = collect_url("https://site.test/kyc", fetcher=page(LONG_TEXT.upper()), store=store, engine=engine)
    assert again["status"] == "duplicate"
    assert len(store.added) == 1 and count_rows(collected_documents, engine) == 1


def test_same_content_under_another_url_is_a_duplicate_too(engine):
    store = RecordingStore()
    collect_url("https://a.test/x", fetcher=page(LONG_TEXT), store=store, engine=engine)
    assert collect_url("https://mirror.test/x", fetcher=page(LONG_TEXT), store=store, engine=engine)["status"] == "duplicate"


def test_changed_page_replaces_the_old_chunks(engine):
    store = RecordingStore()
    collect_url("https://site.test/kyc", fetcher=page(LONG_TEXT), store=store, engine=engine)
    updated = collect_url("https://site.test/kyc", fetcher=page(LONG_TEXT + " New regulation added."), store=store, engine=engine)
    assert updated["status"] == "updated"
    assert store.removed == [{"source": updated["doc_name"]}]
    assert len(store.added) == 2


def test_error_pages_are_not_ingested(engine):
    store = RecordingStore()
    assert collect_url("https://site.test/e", fetcher=page("Not found"), store=store, engine=engine)["status"] == "empty"
    assert not store.added and count_rows(collected_documents, engine) == 0


def test_doc_name_is_a_stable_readable_slug():
    assert doc_name_for("https://en.wikipedia.org/wiki/Know_your_customer", "text/html") == "en_wikipedia_org_wiki_know_your_customer.html"
    assert doc_name_for("https://x.test/a/b.pdf", "application/pdf") == "x_test_a_b_pdf.pdf"


class WordBagEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        vector = [0.0] * 16
        for word in text.lower().split():
            vector[hash(word) % 16] += 1.0
        return vector


def test_real_chroma_upsert_and_removal_are_idempotent(tmp_path):
    from langchain_chroma import Chroma

    from src.ingest import split_documents

    store = Chroma(collection_name="collector_test", embedding_function=WordBagEmbeddings(), persist_directory=str(tmp_path))
    chunks = split_documents([Document(page_content=LONG_TEXT, metadata={"source": "doc.html", "page": 1})])
    add_chunks(store, chunks)
    add_chunks(store, chunks)          # same ids -> no duplicates
    assert store._collection.count() == len(chunks)
    remove_document(store, "doc.html")
    assert store._collection.count() == 0


# --- Celery ---------------------------------------------------------------------------------

@pytest.fixture
def eager():
    tasks.app.conf.task_always_eager = True
    tasks.app.conf.task_store_eager_result = False
    yield
    tasks.app.conf.task_always_eager = False


def test_collect_all_queues_one_task_per_source(eager, monkeypatch):
    seen = []
    monkeypatch.setattr(tasks, "collect_url", lambda url: seen.append(url) or {"url": url, "status": "ingested"})
    count = tasks.collect_all.delay().get()
    sources = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))["sources"]
    assert count == len(sources) and seen == [s["url"] for s in sources]


def test_robots_block_is_reported_not_retried(eager, monkeypatch):
    def blocked(url):
        raise DisallowedByRobots(url)

    monkeypatch.setattr(tasks, "collect_url", blocked)
    result = tasks.collect_one.delay("https://site.test/x").get()
    assert result == {"url": "https://site.test/x", "status": "skipped", "reason": "DisallowedByRobots"}


def test_network_errors_trigger_celery_retries():
    assert requests.RequestException in tasks.collect_one.autoretry_for
    assert tasks.collect_one.max_retries == 3 and tasks.collect_one.retry_jitter is True


def test_beat_schedule_runs_collect_all_periodically():
    entry = tasks.app.conf.beat_schedule["collect-all-sources"]
    assert entry["task"] == "src.collector.tasks.collect_all" and entry["schedule"] > 0


def test_task_is_acknowledged_late_so_a_crash_does_not_lose_it():
    assert tasks.app.conf.task_acks_late is True


@pytest.fixture
def redis_available():
    import redis

    try:
        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1).ping()
    except Exception:
        pytest.skip("Redis is not reachable (docker compose --profile full up -d redis)")


@pytest.mark.redis
def test_task_runs_through_a_real_redis_broker_and_worker(redis_available, monkeypatch):
    from celery.contrib.testing.worker import start_worker

    tasks.app.conf.task_always_eager = False
    with start_worker(tasks.app, pool="solo", perform_ping_check=False, loglevel="WARNING"):
        # collect_all only enqueues work, so it needs no network or vector store here.
        monkeypatch.setattr(tasks, "load_sources", lambda: [])
        assert tasks.collect_all.delay().get(timeout=20) == 0
