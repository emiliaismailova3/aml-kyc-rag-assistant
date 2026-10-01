"""Base RAG (retrieval-augmented generation) pipeline for the AML/KYC
knowledge assistant.

Retrieves the top-k most relevant chunks from the Chroma vector store and
asks the configured LLM (OpenAI-compatible: Groq / Together / OpenAI,
selected via LLM_PROVIDER in .env) to answer strictly from that context.
The system prompt explicitly forbids answering from outside knowledge and
requires an explicit "I don't know" when the context is insufficient, so the
assistant doesn't hallucinate compliance advice.

Usage:
    python -m src.rag "What is beneficial ownership?"
    python -m src.rag --run-eval   # answer every question in data/eval_questions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import PROJECT_ROOT, LLMConfig, get_llm_config
from src.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4

# The exact refusal phrase the system prompt instructs the model to use.
# Checked as a prefix (case-insensitive) since models occasionally vary
# punctuation/casing slightly.
REFUSAL_PREFIX = "i don't know"

# Some models (observed with gpt-oss on Groq) emit built-in-browser-style
# citations like "【3†Source: file.pdf, p. 12】" instead of the requested
# "[3]" style. This cleans those up as a fallback regardless of how well the
# prompt instruction below is followed.
_WEIRD_CITATION_RE = re.compile(r"【\s*(\d+)[^】]*】")

# Rate limits (HTTP 429) are retried automatically by the openai client itself
# (exponential backoff) when constructed with max_retries > its default of 2 --
# bumped here since batch evaluation (Step 8) makes many calls in a row.
LLM_MAX_RETRIES = 5

SYSTEM_PROMPT = """You are an AML/KYC compliance assistant for a neobank/fintech. \
You answer questions using ONLY the excerpts provided in the CONTEXT below, which \
come from FATF, Wolfsberg Group, and Central Bank of Azerbaijan (CBAR) source \
documents.

Rules:
1. Base your answer strictly on the CONTEXT. Do not use outside knowledge, and do \
not guess or extrapolate beyond what the CONTEXT supports.
2. If the CONTEXT does not contain enough information to answer the question, \
respond exactly with: "I don't know based on the available documents." Do not \
attempt a partial or speculative answer in that case.
3. When you do answer, cite the numbered context chunk(s) you used with plain \
bracketed numbers matching the [1], [2], ... labels in the CONTEXT above (e.g. \
"beneficial owners must be identified [1]"). Do not use any other citation \
format (no special brackets, footnote markers, or inline source names).
4. Be precise and factual. This is used for regulatory compliance, so do not \
soften, simplify away, or invent obligations, thresholds, or deadlines.
"""

USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

Answer the question using only the CONTEXT above, following the rules in the \
system prompt."""


class RAGResult(TypedDict):
    question: str
    answer: str
    sources: list[dict]


def format_context(chunks: list[Document]) -> str:
    """Render retrieved chunks into a numbered, source-labeled context block."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        parts.append(f"[{i}] (Source: {source}, p. {page})\n{chunk.page_content}")
    return "\n\n".join(parts)


def is_refusal(answer: str) -> bool:
    """True if the model's answer is the "I don't know" refusal instructed by
    the system prompt, rather than a real answer grounded in retrieved context."""
    return answer.strip().lower().startswith(REFUSAL_PREFIX)


def clean_citations(answer: str) -> str:
    """Normalize non-standard citation formats some models emit (observed:
    gpt-oss's built-in-browser-style "【3†Source: ..., p. 12】") down to the
    plain "[3]" style requested in the prompt."""
    return _WEIRD_CITATION_RE.sub(r"[\1]", answer)


def get_llm(config: LLMConfig | None = None, max_tokens: int | None = None) -> ChatOpenAI:
    """Build the chat LLM client. Pass an explicit `config` (e.g. from
    src.config.get_eval_llm_config()) to use a different model than the one
    that answers questions -- used by src.evaluate for the RAGAS judge.

    `max_tokens` is left at the provider default (None) for normal Q&A, but
    src.evaluate passes a higher explicit value for the RAGAS judge: reasoning
    models like gpt-oss spend part of their output budget on internal
    reasoning before the final answer, and ragas's default expectations can
    hit that cap and raise LLMDidNotFinishException ("generation was not
    completed") if it's too low.
    """
    config = config or get_llm_config()
    if not config.api_key:
        raise RuntimeError(
            f"LLM_PROVIDER={config.provider!r} requires an API key. "
            f"Set it in .env (see .env.example) before calling the LLM."
        )
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.api_base,
        temperature=0,
        max_retries=LLM_MAX_RETRIES,
        max_tokens=max_tokens,
    )


class RAGPipeline:
    """Retrieval + generation over the AML/KYC Chroma knowledge base."""

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self.vectorstore = load_vectorstore()
        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        return self.vectorstore.similarity_search(question, k=k or self.top_k)

    def answer(self, question: str, k: int | None = None) -> RAGResult:
        chunks = self.retrieve(question, k=k)
        context = format_context(chunks)
        messages = [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE.format(context=context, question=question)),
        ]
        response = self.llm.invoke(messages)
        answer_text = clean_citations(response.content)
        # Don't show sources alongside a refusal -- listing the retrieved
        # chunks next to "I don't know" wrongly implies they were used.
        sources = (
            []
            if is_refusal(answer_text)
            else [{"source": c.metadata.get("source"), "page": c.metadata.get("page")} for c in chunks]
        )
        return {"question": question, "answer": answer_text, "sources": sources}


def run_eval_questions(
    eval_path: Path = PROJECT_ROOT / "data" / "eval_questions.json",
    output_path: Path = PROJECT_ROOT / "data" / "eval_results" / "base_rag_results.json",
    top_k: int = DEFAULT_TOP_K,
    delay_seconds: float = 1.0,
) -> list[RAGResult]:
    """Answer every question in the gold eval set and save the results.

    Requires a working LLM_PROVIDER + API key in .env. Output is consumed by
    src/evaluate.py (Step 8) to compute RAGAS metrics for the base pipeline.

    Resumable: if output_path already has results from a previous (possibly
    crashed, e.g. rate-limited) run, already-answered question ids are
    skipped. Results are saved after every question, not just at the end, so
    a crash never loses more than the question in flight. A small delay
    between questions reduces the chance of hitting Groq's free-tier rate
    limit in the first place.
    """
    with open(eval_path, encoding="utf-8") as f:
        eval_data = json.load(f)

    results: list[RAGResult] = []
    completed_ids: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            results = json.load(f)
        completed_ids = {r["id"] for r in results}
        logger.info("Resuming %s: %d questions already answered", output_path, len(completed_ids))

    pipeline = RAGPipeline(top_k=top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for item in eval_data["questions"]:
        if item["id"] in completed_ids:
            continue
        logger.info("Answering %s: %s", item["id"], item["question"])
        result = pipeline.answer(item["question"])
        result["id"] = item["id"]
        result["ground_truth"] = item["ground_truth"]
        results.append(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        time.sleep(delay_seconds)

    logger.info("Saved %d results to %s", len(results), output_path)
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="A single question to ask")
    parser.add_argument(
        "--run-eval", action="store_true", help="Answer every question in data/eval_questions.json"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.run_eval:
        run_eval_questions(top_k=args.top_k)
        return

    if not args.question:
        parser.error("Provide a question, or use --run-eval")

    pipeline = RAGPipeline(top_k=args.top_k)
    result = pipeline.answer(args.question)
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['source']} (p. {s['page']})")


if __name__ == "__main__":
    _main()
