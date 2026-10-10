# Status: what the CV says vs. what the code does

Branch `feature/document-assistant`. "Proved by" lists the tests/commands that back each claim; "Not verified"
says plainly what was **not** run. Numbers come from `data/eval_results/*.json` or from the commands shown.

| CV bullet | Status | Files | Proved by | 1-minute demo |
|---|---|---|---|---|
| **Document Q&A with citations**: RAG with semantic search in PostgreSQL/pgvector, evaluated on a hand-built Q&A set | **Done** (pgvector backend); evaluation ran on the Chroma backend | `src/pgvector_store.py`, `src/db.py`, `src/vectorstore.py`, `data/eval_questions.json` (15 hand-written Q&A) | `tests/test_db.py` (pgvector search + idempotent ingestion, `postgres` marker); live: 2,176 chunks indexed in pgvector with an HNSW index and queried; RAGAS (k=8, 15/15 scored): faithfulness 0.831, answer relevancy 0.840, answer correctness 0.597 | `docker compose --profile full up -d postgres`, `VECTOR_BACKEND=pgvector python -m src.rag "What does beneficial ownership mean?"` |
| **Invoice → structured data**: OCR (Tesseract) + LLM extraction of company, amount, dates, validated with Pydantic | **Done** (synthetic invoices; real Tesseract 5.4 with `eng+aze` ran on all 10 scans/images) | `src/invoices/{ocr,extract,schema,evaluate}.py`, `scripts/generate_synthetic_invoices.py`, `data/invoices/` | `tests/test_invoices.py` (23 tests: business rules, repair loop, deskew); live run on all 15 invoices (`data/eval_results/invoice_extraction.json`): VOEN, invoice number, date, total and currency **100%**; company name **86.7%** (13/15), the 2 misses are OCR dropping diacritics ("Ticarət"→"Ticaret", "Kəpəz"→"Kepez") and the matcher still resolves both to the right company | `python -m src.invoices.evaluate` (needs Tesseract) |
| …company names matched to a reference DB (fuzzy + embedding similarity) | **Done** (synthetic data) | `src/matching/` , `data/reference/` | `tests/test_matching.py` (19 tests); 52 labeled pairs: precision 1.0, recall 1.0, 3 near-miss names routed to human review (`data/eval_results/matching.json`) | `python -m src.matching.evaluate` |
| **Automated data collection**: scraping/parsing web pages and PDFs, cleaning, normalization, dedup, scheduled jobs (Celery + Redis) | **Done locally; Docker worker/beat not run end to end** | `src/collector/`, `data/sources.json`, `docker-compose.yml` (worker, beat) | `tests/test_collector.py` (24 tests incl. a real Redis broker + in-process Celery worker); live: worker collected 2 Wikipedia pages, a second run returned `duplicate` for both and the chunk count did not grow | `celery -A src.collector.tasks worker --pool=solo` then `python -c "from src.collector.tasks import collect_all; print(collect_all.delay().get())"` |
| **AI agent**: function calling across document search, SQL on PostgreSQL, invoice extraction | **Done** | `src/agent.py`, `src/sql_tool.py`, `scripts/check_agent_routing.py`, `data/agent_test_scenarios.json` (t11–t16) | `tests/test_sql_tool.py` (39 tests, 23 attack-style queries rejected, read-only role on real Postgres); live routing: **6/6** scenarios on SQLite, **3/3** SQL scenarios on PostgreSQL with the read-only role, answers matched ground truth | `python -m scripts.seed_demo_db && python -m scripts.check_agent_routing t11 t14` |
| …accepts text **or voice** (Whisper) | **Done, accuracy only spot-checked** | `src/voice.py`, `POST /ask_voice` | `tests/test_voice.py` (11 tests, Whisper mocked); live, 2 real recordings of the author (WhatsApp audio, English): "What is 450 times 37?" transcribed exactly and answered **16,650** via the calculator tool; "What does beneficial ownership mean?" was heard as "beneficial over-ship" by both `base` and `small`, yet the agent still answered correctly from the knowledge base. Two recordings are a spot check, **not** a measured error rate | `curl -F "file=@q.m4a" -F "language=en" localhost:8000/ask_voice` |
| **Production reliability**: latency + token-cost logging | **Done** | `src/llm_client.py`, `src/llm_stats.py`, `src/llm_callbacks.py`, `GET /stats` | `tests/test_llm_client.py` (33 tests); live Groq call logged with tokens and cost | `curl localhost:8000/stats` |
| …routing of simple queries to a smaller model | **Done** (rule-based, not learned) | `choose_tier` in `src/llm_client.py` | unit tests; live: a short prompt went to `openai/gpt-oss-20b` | `python -c "from src.llm_client import get_client; print(get_client().complete([{'role':'user','content':'ping'}]).model)"` |
| …retry with backoff | **Done** | `is_retryable`, `backoff_delay` | unit tests (429/5xx retried, 400/401 not; jitter bounds) | — |
| …OpenAI → Anthropic fallback | **Partial: mocks only** | `LLMClient`, `get_provider_chain` | `tests/test_llm_client.py` fallback tests with scripted providers. **No OpenAI or Anthropic key was available**, so neither provider was called live; only Groq was | set `LLM_FALLBACK_CHAIN=openai,anthropic,groq` plus the keys |
| …human escalation | **Done** | `LLMUnavailable`, API handler, `needs_human_review` | API test: 503 + `escalate_to_human: true`; invoice extraction returns `needs_human_review` | — |

## What is NOT done or NOT verified (read this before the interview)

- **OCR was measured only on synthetic, single-template invoices**: 5 clean PNGs, 3 noisy JPGs and 2 image-only PDFs generated by `scripts/generate_synthetic_invoices.py` with a clean font. Real photos, stamps, handwriting and varied layouts are untested, and the noisy set scoring 100% mostly shows that the synthetic noise is mild.
- **Invoice numbers are not real-world accuracy**: the 100% fields mean the pipeline works on this template, nothing more.
- **The agent has no provider fallback**: it uses LangChain's `ChatOpenAI` for tool calling. It is logged by a callback and
  uses the SDK's built-in retries only.
- **The RAGAS table was not re-run on pgvector and the agent was not re-scored** after the new tools were added. The agent
  RAGAS evaluation was started on `main` (3 tools, 4 chunks per search) and stopped by Groq's daily quota; its partial
  cache is from that agent, so re-run it from scratch on whichever branch you present.
- **Voice quality is only spot-checked** (2 recordings, no WER): one phrase was exact, one word was misheard by both Whisper sizes. Azerbaijani speech was not tested.
- **Docker:** the image builds with all new dependencies (built locally, and the CI `docker` job that boots API + UI is green on this
  branch). The `worker` and `beat` *containers* were never started: only `docker compose config` validation, plus a worker run directly on the host.
- Matching and invoices use **synthetic fictional data**; nothing here was tested on real company documents.
- The demo invoice rows in the database come from the ground-truth labels, not from the extractor.

## Test suite

`pytest -q -k "not internet_search"` with PostgreSQL and Redis running: **225 passed**, 1 deselected (the live web-search test). `ruff check src tests scripts`: clean.
Without those services the `postgres`/`redis`-marked tests skip themselves.
