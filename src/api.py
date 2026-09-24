"""FastAPI backend for the AML/KYC knowledge assistant.

Exposes POST /ask, which takes a question, retrieves relevant chunks from
the Chroma knowledge base, asks the configured LLM to answer strictly from
that context, and returns the answer with its cited sources. Every request
is logged to a local SQLite database (src/logging_db.py) for later review.

Run with:
    uvicorn src.api:app --reload
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import AgentPipeline
from src.logging_db import log_request
from src.rag import RAGPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AML/KYC Knowledge Assistant API",
    description=(
        "Retrieval-augmented question answering over AML/KYC compliance "
        "documents (FATF, Wolfsberg Group, CBAR)."
    ),
    version="0.1.0",
)

# The pipeline loads the embedding model and opens the Chroma store once at
# import time (module-level singleton) so each request doesn't pay that cost.
_pipeline: RAGPipeline | None = None
_agent_pipeline: AgentPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def get_agent_pipeline() -> AgentPipeline:
    global _agent_pipeline
    if _agent_pipeline is None:
        _agent_pipeline = AgentPipeline()
    return _agent_pipeline


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The compliance question to ask")
    top_k: int = Field(4, ge=1, le=10, description="Number of chunks to retrieve")


class SourceRef(BaseModel):
    source: str
    page: int | None = None


class ToolCall(BaseModel):
    tool: str
    input: dict | str
    output: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef] = []
    tool_calls: list[ToolCall] = []
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    pipeline = get_pipeline()
    start = time.perf_counter()
    try:
        result = pipeline.answer(request.question, k=request.top_k)
    except RuntimeError as exc:
        # Raised by src.rag.get_llm() when no API key is configured -- surface
        # that as a clear 503 rather than a generic 500.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - start) * 1000

    log_request(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        pipeline="rag",
        latency_ms=latency_ms,
    )

    return AskResponse(answer=result["answer"], sources=result["sources"], latency_ms=latency_ms)


@app.post("/ask_agent", response_model=AskResponse)
def ask_agent(request: AskRequest) -> AskResponse:
    """Same contract as /ask, but routed through the tool-calling agent
    (src.agent.AgentPipeline), which may call the knowledge base, a
    calculator, and/or web search before answering."""
    pipeline = get_agent_pipeline()
    start = time.perf_counter()
    try:
        result = pipeline.answer(request.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - start) * 1000

    log_request(
        question=request.question,
        answer=result["answer"],
        sources=result["tool_calls"],
        pipeline="agent",
        latency_ms=latency_ms,
    )

    return AskResponse(
        answer=result["answer"],
        sources=[],
        tool_calls=result["tool_calls"],
        latency_ms=latency_ms,
    )
