# AI Knowledge Assistant — AML/KYC Compliance RAG

> **Status: work in progress.** This README is a planning stub for Step 1 of the build.
> It will be replaced with the full architecture, setup instructions, examples, and
> evaluation results at the end of the project (final step).

## What this is

A retrieval-augmented generation (RAG) question-answering service over AML/KYC
(Anti-Money Laundering / Know Your Customer) and financial compliance documents for a
neobank/fintech, with:

- An **agentic layer** (tool-calling) that can decide between answering from the
  knowledge base, using a calculator tool, or using a web search tool.
- A **FastAPI** backend (`POST /ask`) with request logging.
- A **Streamlit** demo UI.
- A **RAGAS**-based evaluation framework (relevance, faithfulness, answer correctness),
  comparing the base RAG pipeline against the agentic pipeline.
- **Docker** packaging for the API + UI.

Built as a portfolio project for a Junior AI Applications & AI Agents Engineer role.

## Domain & corpus

Source documents (FATF recommendations, FATF Azerbaijan mutual evaluation follow-up,
Wolfsberg Group guidance, CBAR AML/payment institution regulations) live in
[`data/raw/`](data/raw/). See [`data/raw/SOURCES.md`](data/raw/SOURCES.md) for the
full list with original URLs and retrieval dates.

## Planned architecture

```
data/raw/*.pdf ──▶ src/ingest.py ──▶ chunks + metadata ──▶ Chroma vector store
                                                                   │
User question ──▶ src/rag.py (retrieval top-k + LLM) ◀────────────┘
                        │
                        ▼
            src/agent.py (tool-calling: KB retrieval / calculator / web search)
                        │
                        ▼
              src/api.py (FastAPI: POST /ask) ──▶ Streamlit UI
                        │
                        ▼
              src/evaluate.py (RAGAS metrics on data/eval_questions.json)
```

A full mermaid diagram will be added to this README in the final step.

## Build plan (executed as staged commits)

1. ✅ Repo scaffolding, `.gitignore`, `requirements.txt`, source corpus download.
2. ✅ Gold evaluation question set (`data/eval_questions.json`).
3. ✅ Document ingestion pipeline (`src/ingest.py`).
4. ✅ Embeddings + Chroma vector store + retrieval smoke test.
5. ✅ Base RAG pipeline (`src/rag.py`).
6. ✅ FastAPI backend + Streamlit demo UI.
7. ✅ Agentic tool-calling layer (calculator + web search tools).
8. ⬜ RAGAS evaluation: base RAG vs. agentic layer.
9. ⬜ Docker + docker-compose packaging.
10. ⬜ Final README: architecture diagram, usage, examples, metrics, CV bullets.

## Quickstart (placeholder — will be finalized in step 10)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env       # fill in your API key(s)
```

## Tech stack

Python 3.11+, LangChain, ChromaDB, OpenAI API (swappable for Groq/Together via
`.env`), FastAPI, Streamlit, Docker, RAGAS.

## License

TBD.
