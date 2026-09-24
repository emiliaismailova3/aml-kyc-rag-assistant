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
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import PROJECT_ROOT, get_llm_config
from src.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4

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
3. When you do answer, cite the source document(s) you used, e.g. \
"(Source: fatf_recommendations_2025.pdf, p. 14)".
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


def get_llm() -> ChatOpenAI:
    config = get_llm_config()
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
        sources = [
            {"source": c.metadata.get("source"), "page": c.metadata.get("page")}
            for c in chunks
        ]
        return {"question": question, "answer": response.content, "sources": sources}


def run_eval_questions(
    eval_path: Path = PROJECT_ROOT / "data" / "eval_questions.json",
    output_path: Path = PROJECT_ROOT / "data" / "eval_results" / "base_rag_results.json",
    top_k: int = DEFAULT_TOP_K,
) -> list[RAGResult]:
    """Answer every question in the gold eval set and save the results.

    Requires a working LLM_PROVIDER + API key in .env. Output is consumed by
    src/evaluate.py (Step 8) to compute RAGAS metrics for the base pipeline.
    """
    with open(eval_path, encoding="utf-8") as f:
        eval_data = json.load(f)

    pipeline = RAGPipeline(top_k=top_k)
    results = []
    for item in eval_data["questions"]:
        logger.info("Answering %s: %s", item["id"], item["question"])
        result = pipeline.answer(item["question"])
        result["id"] = item["id"]
        result["ground_truth"] = item["ground_truth"]
        results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
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
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (p. {s['page']})")


if __name__ == "__main__":
    _main()
