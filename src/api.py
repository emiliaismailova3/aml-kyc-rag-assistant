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
import tempfile
import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from openai import APIError, RateLimitError
from pydantic import BaseModel, Field

from src.agent import AgentPipeline
from src.invoices.extract import ExtractionResult, extract_invoice
from src.invoices.ocr import IMAGE_SUFFIXES
from src.llm_client import LLMUnavailable
from src.llm_stats import get_stats
from src.logging_db import log_request
from src.rag import RAGPipeline
from src.voice import AUDIO_SUFFIXES, MAX_AUDIO_BYTES, VoiceUnavailable, transcribe

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
    top_k: int = Field(8, ge=1, le=12, description="Number of chunks to retrieve")


class SourceRef(BaseModel):
    ref: int | None = Field(None, description="Citation number used in the answer text, e.g. 3 for [3]")
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


def _llm_http_error(exc: Exception) -> HTTPException:
    """Map failures of the LLM provider to a clear JSON error instead of a bare 500."""
    if isinstance(exc, RateLimitError):
        return HTTPException(
            status_code=429,
            detail="The LLM provider's rate limit or daily token quota has been reached. "
            "Please try again later.",
        )
    return HTTPException(status_code=502, detail=f"The LLM provider returned an error: {exc}")


@app.exception_handler(LLMUnavailable)
async def llm_unavailable_handler(request: Request, exc: LLMUnavailable) -> JSONResponse:
    """Every provider in the fallback chain failed: tell the client, and flag it for a human."""
    return JSONResponse(
        status_code=429 if exc.rate_limited else 503,
        content={"detail": str(exc), "escalate_to_human": True},
    )


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
    except APIError as exc:
        raise _llm_http_error(exc) from exc
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
    except APIError as exc:
        raise _llm_http_error(exc) from exc
    # The agent swallows provider errors into a fallback message; surface a
    # transient one as an error instead of a fake 200 answer. A step-limit loop
    # (GraphRecursionError) is real agent behaviour and is returned as-is.
    if result.get("error") and result["error"] != "GraphRecursionError":
        raise HTTPException(
            status_code=429 if "RateLimit" in result["error"] else 502,
            detail="The agent could not reach the LLM provider "
            f"({result['error']}); the daily token quota may be exhausted. Please try again later.",
        )
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
        sources=result.get("sources", []),
        tool_calls=result["tool_calls"],
        latency_ms=latency_ms,
    )


@app.post("/invoices/extract", response_model=ExtractionResult)
async def extract_invoice_endpoint(file: Annotated[UploadFile, File()]) -> ExtractionResult:
    """Upload an invoice (PDF or image); get back validated structured data,
    or needs_human_review=true if the LLM could not produce a valid result."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".pdf" and suffix not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=415, detail="Upload a PDF or an image (png, jpg, tiff).")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(await file.read())
        try:
            return extract_invoice(path)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except APIError as exc:
            raise _llm_http_error(exc) from exc
        except Exception as exc:  # noqa: BLE001 - OCR engine missing etc.
            if type(exc).__name__ == "TesseractNotFoundError":
                raise HTTPException(
                    status_code=503, detail="OCR is unavailable: the Tesseract binary is not installed."
                ) from exc
            raise


class VoiceResponse(AskResponse):
    transcript: str


@app.post("/ask_voice", response_model=VoiceResponse)
async def ask_voice(
    file: Annotated[UploadFile, File()], language: Annotated[str | None, Form()] = None
) -> VoiceResponse:
    """Spoken question: audio -> Whisper -> the same agent as /ask_agent.
    `language` is an optional ISO code such as "en" or "az"; by default Whisper detects it."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Upload audio in one of: {', '.join(sorted(AUDIO_SUFFIXES))}")
    audio = await file.read()
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio is larger than 25 MB.")
    try:
        transcript = await run_in_threadpool(transcribe, audio, file.filename or f"audio{suffix}", language)
    except VoiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not transcript:
        raise HTTPException(status_code=422, detail="No speech could be recognised in the audio.")

    answer = await run_in_threadpool(ask_agent, AskRequest(question=transcript))
    return VoiceResponse(**answer.model_dump(), transcript=transcript)


@app.get("/stats")
def stats() -> dict:
    """Latency percentiles, error rate and estimated cost of the recorded LLM calls."""
    return get_stats()
