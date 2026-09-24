"""RAGAS-based quality evaluation for the AML/KYC knowledge assistant (Step 8).

Runs the gold question set (data/eval_questions.json) through both the base
RAG pipeline (src.rag.RAGPipeline) and the agentic pipeline (src.agent.AgentPipeline),
computes RAGAS metrics for each, and writes a side-by-side comparison so the
effect of adding the agentic tool-calling layer on answer quality can be seen
directly (Step 8 deliverable: "before agent / after agent" metrics table).

Metrics computed:
  - faithfulness: does the answer avoid claims unsupported by the retrieved context?
  - answer_relevancy: does the answer actually address the question asked?
  - answer_correctness: how close is the answer to the gold ground_truth?
  - context_precision / context_recall: retrieval quality against the ground truth.

Requires a working LLM_PROVIDER + API key in .env (RAGAS uses the LLM as a
judge for faithfulness/relevancy/correctness). Local embeddings are reused
for the embedding-based metrics, so no OpenAI key is required for that part.

Usage:
    python -m src.evaluate --pipeline rag
    python -m src.evaluate --pipeline agent
    python -m src.evaluate --pipeline both   # also writes the comparison table
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

EVAL_QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval_questions.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval_results"

RAGAS_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
]


def _load_eval_questions() -> list[dict]:
    with open(EVAL_QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)["questions"]


def _collect_rag_samples(questions: list[dict]) -> list[dict]:
    """Run every question through the base RAG pipeline, capturing the
    retrieved context text (not just source labels) for RAGAS's context
    metrics."""
    from src.rag import RAGPipeline

    pipeline = RAGPipeline()
    samples = []
    for item in questions:
        chunks = pipeline.retrieve(item["question"])
        result = pipeline.answer(item["question"])
        samples.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": result["answer"],
                "contexts": [c.page_content for c in chunks],
                "ground_truth": item["ground_truth"],
            }
        )
        logger.info("[rag] answered %s", item["id"])
    return samples


def _collect_agent_samples(questions: list[dict]) -> list[dict]:
    """Run every question through the agentic pipeline. Contexts are taken
    from any search_knowledge_base tool call the agent made; if it didn't
    consult the knowledge base (e.g. it used the calculator instead), the
    contexts list is empty, which RAGAS's context metrics naturally penalize
    -- appropriate, since those questions are RAG-answerable and the ground
    truth in data/eval_questions.json is always knowledge-base-derived."""
    from src.agent import AgentPipeline

    pipeline = AgentPipeline()
    samples = []
    for item in questions:
        result = pipeline.answer(item["question"])
        contexts = [
            call["output"]
            for call in result["tool_calls"]
            if call["tool"] == "search_knowledge_base"
        ]
        samples.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": result["answer"],
                "contexts": contexts or [""],  # RAGAS requires a non-empty contexts list
                "ground_truth": item["ground_truth"],
            }
        )
        logger.info("[agent] answered %s", item["id"])
    return samples


def _patch_ragas_vertexai_import() -> None:
    """Work around a broken transitive import in installed ragas versions.

    ragas.llms.base unconditionally does
    `from langchain_community.chat_models.vertexai import ChatVertexAI`,
    but that submodule was removed from langchain-community in the version
    this project otherwise depends on (langchain-community>=0.4, required
    for compatibility with LangChain 1.x). We never use Vertex AI, so a
    minimal stub module satisfies the import without pulling in Google Cloud
    dependencies or downgrading langchain-community (which would break
    LangChain 1.x compatibility across the rest of the project).
    """
    import sys
    import types

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    shim = types.ModuleType(module_name)
    shim.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[module_name] = shim


def _run_ragas(samples: list[dict]) -> dict:
    """Compute RAGAS metrics for a list of {question, answer, contexts,
    ground_truth} samples, using our configured LLM and local embeddings as
    the judge/embedder so no OpenAI key is required beyond the chat LLM."""
    _patch_ragas_vertexai_import()

    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from src.rag import get_llm
    from src.vectorstore import get_embeddings

    dataset = Dataset.from_list(
        [
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "ground_truth": s["ground_truth"],
            }
            for s in samples
        ]
    )

    ragas_llm = LangchainLLMWrapper(get_llm())
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result.to_pandas().to_dict(orient="records"), {
        name: float(result[name]) for name in RAGAS_METRIC_NAMES if name in result
    }


def evaluate_pipeline(pipeline_name: str) -> dict:
    """Evaluate one pipeline ("rag" or "agent") and save its results."""
    questions = _load_eval_questions()

    if pipeline_name == "rag":
        samples = _collect_rag_samples(questions)
    elif pipeline_name == "agent":
        samples = _collect_agent_samples(questions)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name!r}")

    per_question, aggregate = _run_ragas(samples)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{pipeline_name}_ragas_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"per_question": per_question, "aggregate": aggregate}, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s RAGAS results to %s", pipeline_name, output_path)
    logger.info("%s aggregate metrics: %s", pipeline_name, aggregate)
    return aggregate


def write_comparison_table(rag_metrics: dict, agent_metrics: dict) -> Path:
    """Write a Markdown table comparing the two pipelines, ready to paste
    into README.md (Step 10)."""
    lines = [
        "| Metric | Base RAG | Agentic (RAG + tools) |",
        "|---|---|---|",
    ]
    for name in RAGAS_METRIC_NAMES:
        rag_val = rag_metrics.get(name)
        agent_val = agent_metrics.get(name)
        rag_str = f"{rag_val:.3f}" if rag_val is not None else "n/a"
        agent_str = f"{agent_val:.3f}" if agent_val is not None else "n/a"
        lines.append(f"| {name} | {rag_str} | {agent_str} |")

    table_md = "\n".join(lines) + "\n"
    output_path = RESULTS_DIR / "metrics_comparison.md"
    output_path.write_text(table_md, encoding="utf-8")
    logger.info("Wrote comparison table to %s", output_path)
    return output_path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=["rag", "agent", "both"], default="both")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rag_metrics = agent_metrics = None
    if args.pipeline in ("rag", "both"):
        rag_metrics = evaluate_pipeline("rag")
    if args.pipeline in ("agent", "both"):
        agent_metrics = evaluate_pipeline("agent")

    if args.pipeline == "both" and rag_metrics and agent_metrics:
        path = write_comparison_table(rag_metrics, agent_metrics)
        print(f"\nComparison table written to {path}")


if __name__ == "__main__":
    _main()
