# Status: what the CV says vs. what the code does

Branch `feature/document-assistant`. "Proved by" lists the tests/commands that back each claim; "Not verified"
says plainly what was **not** run. Numbers come from `data/eval_results/*.json` or from the commands shown.

| CV bullet | Status | Files | Proved by | 1-minute demo |
|---|---|---|---|---|
| **Document Q&A with citations**: RAG with semantic search in PostgreSQL/pgvector, evaluated on a hand-built Q&A set | **Done** (pgvector backend); evaluation ran on the Chroma backend | `src/pgvector_store.py`, `src/db.py`, `src/vectorstore.py`, `data/eval_questions.json` (15 hand-written Q&A) | `tests/test_db.py` (pgvector search + idempotent ingestion, `postgres` marker); live: 2,176 chunks indexed in pgvector with an HNSW index and queried; RAGAS (k=8, 15/15 scored): faithfulness 0.831, answer relevancy 0.840, answer correctness 0.597 | `docker compose --profile full up -d postgres`, `VECTOR_BACKEND=pgvector python -m src.rag "What does beneficial ownership mean?"` |
| **Invoice → structured data**: OCR (Tesseract) + LLM extraction of company, amount, dates, validated with Pydantic | **Partial**: extraction + validation done and measured on text-layer PDFs; **OCR path not run on real scans** | `src/invoices/{ocr,extract,schema,evaluate}.py`, `scripts/generate_synthetic_invoices.py`, `data/invoices/` | `tests/test_invoices.py` (22 tests: business rules, repair loop, deskew); live LLM run: **5/5 text-layer PDFs correct on all 6 fields** (`data/eval_results/invoice_extraction.json`) | `python -m src.invoices.evaluate --kinds text` |
| …company names matched to a reference DB (fuzzy + embedding similarity) | **Done** (synthetic data) | `src/matching/` , `data/reference/` | `tests/test_matching.py` (19 tests); 52 labeled pairs: precision 1.0, recall 1.0, 3 near-miss names routed to human review (`data/eval_results/matching.json`) | `python -m src.matching.evaluate` |
| **Automated data collection**: scraping/parsing web pages and PDFs, cleaning, normalization, dedup, scheduled jobs (Celery + Redis) | **Done locally; Docker worker/beat not run end to end** | `src/collector/`, `data/sources.json`, `docker-compose.yml` (worker, beat) | `tests/test_collector.py` (24 tests incl. a real Redis broker + in-process Celery worker); live: worker collected 2 Wikipedia pages, a second run returned `duplicate` for both and the chunk count did not grow | `celery -A src.collector.tasks worker --pool=solo` then `python -c "from src.collector.tasks import collect_all; print(collect_all.delay().get())"` |
| **AI agent**: function calling across document search, SQL on PostgreSQL, invoice extraction | **Done** | `src/agent.py`, `src/sql_tool.py`, `scripts/check_agent_routing.py`, `data/agent_test_scenarios.json` (t11–t16) | `tests/test_sql_tool.py` (39 tests, 23 attack-style queries rejected, read-only role on real Postgres); live routing: **6/6** scenarios on SQLite, **3/3** SQL scenarios on PostgreSQL with the read-only role, answers matched ground truth | `python -m scripts.seed_demo_db && python -m scripts.check_agent_routing t11 t14` |
| …accepts text **or voice** (Whisper) | **Partial**: pipeline works, **accuracy not measured** | `src/voice.py`, `POST /ask_voice` | `tests/test_voice.py` (11 tests, Whisper mocked); live: synthesized audio → faster-whisper → text (one robotic Windows-TTS sample was misheard by both the `base` and `small` models; the audio itself may be the problem, so record your own voice and check before demoing, see `docs/learn/07-voice.md`) | `curl -F "file=@q.wav" localhost:8000/ask_voice` |
| **Production reliability**: latency + token-cost logging | **Done** | `src/llm_client.py`, `src/llm_stats.py`, `src/llm_callbacks.py`, `GET /stats` | `tests/test_llm_client.py` (33 tests); live Groq call logged with tokens and cost | `curl localhost:8000/stats` |
| …routing of simple queries to a smaller model | **Done** (rule-based, not learned) | `choose_tier` in `src/llm_client.py` | unit tests; live: a short prompt went to `openai/gpt-oss-20b` | `python -c "from src.llm_client import get_client; print(get_client().complete([{'role':'user','content':'ping'}]).model)"` |
| …retry with backoff | **Done** | `is_retryable`, `backoff_delay` | unit tests (429/5xx retried, 400/401 not; jitter bounds) | — |
| …OpenAI → Anthropic fallback | **Partial: mocks only** | `LLMClient`, `get_provider_chain` | `tests/test_llm_client.py` fallback tests with scripted providers. **No OpenAI or Anthropic key was available**, so neither provider was called live; only Groq was | set `LLM_FALLBACK_CHAIN=openai,anthropic,groq` plus the keys |
| …human escalation | **Done** | `LLMUnavailable`, API handler, `needs_human_review` | API test: 503 + `escalate_to_human: true`; invoice extraction returns `needs_human_review` | — |

## What is NOT done or NOT verified (read this before the interview)

- **Tesseract OCR was never run for real.** The binary is not installed on the development machine, so the 10 scanned
  / image invoices were skipped by the evaluation (it says so in the JSON) and the OCR accuracy is unknown.
  Preprocessing (grayscale, deskew, threshold) is tested; recognition is not.
  To run it: install Tesseract (Windows: `winget install UB-Mannheim.TesseractOCR`; add the `aze` language data),
  then `python -m src.invoices.evaluate`.
- **Invoice accuracy of 100% is on 5 clean, templated, synthetic PDFs.** It says the pipeline works, not that it is accurate on real invoices.
- **The agent has no provider fallback**: it uses LangChain's `ChatOpenAI` for tool calling. It is logged by a callback and
  uses the SDK's built-in retries only.
- **The RAGAS table was not re-run on pgvector and the agent was not re-scored** after the new tools were added. The agent
  RAGAS evaluation was started on `main` (3 tools, 4 chunks per search) and stopped by Groq's daily quota; its partial
  cache is from that agent, so re-run it from scratch on whichever branch you present.
- **Voice quality is unmeasured** (no recordings set, no WER). Azerbaijani speech was not tested.
- **Docker:** `worker` and `beat` services are validated with `docker compose config`; see the note at the bottom
  about the image build.
- Matching and invoices use **synthetic fictional data**; nothing here was tested on real company documents.
- The demo invoice rows in the database come from the ground-truth labels, not from the extractor.

## Test suite

`pytest -q -k "not internet_search"` with PostgreSQL and Redis running: **224 passed**, 1 deselected (the live web-search test). `ruff check src tests scripts`: clean.
Without those services the `postgres`/`redis`-marked tests skip themselves.
