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
import math
import time
from pathlib import Path
from typing import Callable

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

EVAL_QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval_questions.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval_results"

# Small delay between questions to stay under Groq's free-tier rate limit
# proactively, on top of the LLM client's own retry-with-backoff on 429s
# (src.rag.LLM_MAX_RETRIES) once a limit is actually hit.
INTER_QUESTION_DELAY_SECONDS = 1.0

# The three metrics the project brief asks for. context_precision/context_recall
# were dropped: each extra metric multiplies the judge-LLM token cost, and with
# all five a single pipeline consumed an entire free-tier daily quota (200k
# tokens on Groq) without finishing.
RAGAS_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]


def _load_eval_questions() -> list[dict]:
    with open(EVAL_QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)["questions"]


def _collect_samples_with_resume(
    questions: list[dict],
    answer_one: Callable[[dict], dict],
    cache_path: Path,
    delay_seconds: float = INTER_QUESTION_DELAY_SECONDS,
) -> list[dict]:
    """Run answer_one(item) for every question not already in cache_path,
    saving the growing sample list to disk after every question.

    This makes `python -m src.evaluate` resumable: a run that dies partway
    through (e.g. the LLM client's retries are finally exhausted on a
    sustained Groq rate limit) can simply be re-run and will pick up where
    it left off instead of re-answering already-completed questions.
    """
    samples: list[dict] = []
    completed_ids: set[str] = set()
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            samples = json.load(f)
        completed_ids = {s["id"] for s in samples}
        logger.info("Resuming %s: %d questions already answered", cache_path, len(completed_ids))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for item in questions:
        if item["id"] in completed_ids:
            continue
        try:
            sample = answer_one(item)
        except Exception:
            logger.exception(
                "Failed to answer %s after retries; %d results saved to %s so far -- "
                "re-run this command to resume from here.",
                item["id"],
                len(samples),
                cache_path,
            )
            raise
        samples.append(sample)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        logger.info("answered %s", item["id"])
        time.sleep(delay_seconds)

    return samples


def _collect_rag_samples(questions: list[dict]) -> list[dict]:
    """Run every question through the base RAG pipeline, capturing the
    retrieved context text (not just source labels) for RAGAS's context
    metrics."""
    from src.rag import RAGPipeline

    pipeline = RAGPipeline()

    def answer_one(item: dict) -> dict:
        chunks = pipeline.retrieve(item["question"])
        result = pipeline.answer(item["question"])
        return {
            "id": item["id"],
            "question": item["question"],
            "answer": result["answer"],
            "contexts": [c.page_content for c in chunks],
            "ground_truth": item["ground_truth"],
        }

    return _collect_samples_with_resume(questions, answer_one, RESULTS_DIR / "rag_samples_cache.json")


def _collect_agent_samples(questions: list[dict]) -> list[dict]:
    """Run every question through the agentic pipeline. Contexts are taken
    from any search_knowledge_base tool call the agent made; if it didn't
    consult the knowledge base (e.g. it used the calculator instead), the
    contexts list is empty, which RAGAS's context metrics naturally penalize
    -- appropriate, since those questions are RAG-answerable and the ground
    truth in data/eval_questions.json is always knowledge-base-derived."""
    from src.agent import AgentPipeline

    pipeline = AgentPipeline()

    def answer_one(item: dict) -> dict:
        result = pipeline.answer(item["question"])
        # GraphRecursionError means the agent looped until its step cap without
        # answering -- genuine agent behaviour, recorded as-is and scored (it
        # will score ~0). Anything else (rate limits, transport errors) is
        # transient and must not be cached as an answer.
        if result.get("error") and result["error"] != "GraphRecursionError":
            # The agent's graceful fallback ("I couldn't complete this request")
            # is not an answer: caching and scoring it would silently poison the
            # results (this happened when the token quota ran out mid-run).
            raise RuntimeError(f"agent failed on {item['id']} ({result['error']}); not caching it")
        contexts = [
            call["output"]
            for call in result["tool_calls"]
            if call["tool"] == "search_knowledge_base"
        ]
        return {
            "id": item["id"],
            "question": item["question"],
            "answer": result["answer"],
            "contexts": contexts or [""],  # RAGAS requires a non-empty contexts list
            "ground_truth": item["ground_truth"],
        }

    return _collect_samples_with_resume(questions, answer_one, RESULTS_DIR / "agent_samples_cache.json")


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


def _build_scorer() -> Callable[[dict], dict]:
    """Build the RAGAS judge/embeddings/metrics once and return
    score_one(sample) -> {metric_name: float | None}.

    Both the judge LLM and the embeddings are explicitly our own configured
    ones (never RAGAS's OpenAI default): the LLM via src.config.get_eval_llm_config()
    (same provider/key as the main LLM, but the model can be overridden with
    EVAL_LLM_MODEL in .env -- useful if the model answering questions, e.g.
    gpt-oss, emits output that breaks RAGAS's JSON-based scoring prompts), and
    embeddings via our local sentence-transformers model, so no
    OPENAI_API_KEY is ever required.

    A metric that ragas could not compute (it swallows per-metric errors such
    as rate limits into NaN rather than raising) comes back as None.
    """
    _patch_ragas_vertexai_import()

    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, answer_correctness, faithfulness
    from ragas.run_config import RunConfig

    from src.config import get_eval_llm_config
    from src.rag import get_llm
    from src.vectorstore import get_embeddings

    # max_tokens is bumped for the judge: gpt-oss (and other reasoning models)
    # spend part of the output budget on internal reasoning before the final
    # JSON answer, and ragas's prompts otherwise hit the provider default cap
    # mid-generation (observed: LLMDidNotFinishException).
    ragas_llm = LangchainLLMWrapper(get_llm(get_eval_llm_config(), max_tokens=4096))
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    # answer_relevancy's default strictness=3 asks the LLM to generate 3
    # reverse-engineered questions in a single call (n=3) to average over for
    # robustness. Groq rejects any n>1 ("'n': number must be at most 1"), so
    # this is pinned to 1 -- a real accuracy/robustness tradeoff (one sampled
    # question instead of three), not a cosmetic workaround.
    metrics = [faithfulness, AnswerRelevancy(strictness=1), answer_correctness]

    # max_workers=1: run RAGAS's own judge-LLM calls sequentially rather than
    # the default 16-way concurrency, since Groq's free tier rate-limits
    # (429) hard under concurrent load. max_retries is kept low (2, vs
    # ragas's default 10): our own LLM client already retries each individual
    # API call up to LLM_MAX_RETRIES (5) times with backoff, so a high
    # ragas-level retry count on top multiplies into dozens of attempts per
    # metric once a request is genuinely failing (e.g. a daily token-quota
    # error, which won't resolve within any backoff window).
    run_config = RunConfig(max_workers=1, max_retries=2, max_wait=30, timeout=120)

    def score_one(sample: dict) -> dict:
        dataset = Dataset.from_list(
            [
                {
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "contexts": sample["contexts"],
                    "ground_truth": sample["ground_truth"],
                }
            ]
        )
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
            show_progress=False,
        )
        row = result.to_pandas().iloc[0]
        scores = {}
        for name in RAGAS_METRIC_NAMES:
            value = row.get(name)
            scores[name] = None if value is None or math.isnan(float(value)) else float(value)
        return scores

    return score_one


def _score_samples_with_resume(
    samples: list[dict], cache_path: Path, max_consecutive_empty: int = 2
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Score each sample with RAGAS one at a time, caching every fully-scored
    question to cache_path so a run cut short (e.g. by a free-tier daily token
    quota -- a single pipeline can use a whole day's allowance) can be
    re-run later and only score what's still missing.

    Only questions where *every* metric computed are cached; a question with a
    missing (NaN) metric is retried on the next run. If several questions in a
    row come back with no metric at all, the provider is almost certainly
    refusing requests (quota exhausted), so scoring stops early instead of
    grinding through doomed retries for every remaining question.

    Returns (complete, incomplete), each mapping question id -> metric scores.
    """
    complete: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            complete = {row["id"]: row["scores"] for row in json.load(f)}
        logger.info("Resuming %s: %d questions already scored", cache_path, len(complete))

    todo = [s for s in samples if s["id"] not in complete]
    incomplete: dict[str, dict] = {}
    if not todo:
        return complete, incomplete

    score_one = _build_scorer()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    consecutive_empty = 0
    for sample in todo:
        scores = score_one(sample)
        if all(v is None for v in scores.values()):
            consecutive_empty += 1
            incomplete[sample["id"]] = scores
            logger.warning("%s: no metric could be computed", sample["id"])
            if consecutive_empty >= max_consecutive_empty:
                logger.error(
                    "%d questions in a row produced no scores -- the LLM provider is "
                    "likely refusing requests (e.g. daily token quota exhausted). "
                    "Stopping; %d questions are cached in %s, re-run later to resume.",
                    consecutive_empty,
                    len(complete),
                    cache_path,
                )
                break
            continue
        consecutive_empty = 0
        if any(v is None for v in scores.values()):
            incomplete[sample["id"]] = scores
            logger.warning("%s: incomplete scores %s -- will be retried next run", sample["id"], scores)
            continue
        complete[sample["id"]] = scores
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([{"id": k, "scores": v} for k, v in complete.items()], f, indent=2)
        logger.info("scored %s: %s", sample["id"], scores)

    return complete, incomplete


def _aggregate(rows: list[dict]) -> dict:
    """Mean of each metric over the questions where it was computed."""
    aggregate = {}
    for name in RAGAS_METRIC_NAMES:
        values = [r[name] for r in rows if r.get(name) is not None]
        if values:
            aggregate[name] = sum(values) / len(values)
    return aggregate


def evaluate_pipeline(pipeline_name: str) -> dict:
    """Evaluate one pipeline ("rag" or "agent") and save its results."""
    questions = _load_eval_questions()

    if pipeline_name == "rag":
        samples = _collect_rag_samples(questions)
    elif pipeline_name == "agent":
        samples = _collect_agent_samples(questions)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name!r}")

    complete, incomplete = _score_samples_with_resume(
        samples, RESULTS_DIR / f"{pipeline_name}_scores_cache.json"
    )
    per_question = [{"id": k, **v} for k, v in complete.items()] + [
        {"id": k, **v, "incomplete": True} for k, v in incomplete.items()
    ]
    aggregate = _aggregate(per_question)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{pipeline_name}_ragas_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "aggregate": aggregate,
                "questions_fully_scored": len(complete),
                "questions_total": len(samples),
                "per_question": per_question,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Saved %s RAGAS results to %s", pipeline_name, output_path)
    logger.info(
        "%s: %d/%d questions fully scored; aggregate metrics: %s",
        pipeline_name,
        len(complete),
        len(samples),
        aggregate,
    )
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
