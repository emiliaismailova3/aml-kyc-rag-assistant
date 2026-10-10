"""Central place for reading configuration from environment variables (.env).

Keeping this in one module means every other module (vectorstore, rag, agent,
api) reads settings the same way and it's obvious what needs to be set in
.env for the project to run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# httpx/huggingface_hub log every HEAD/GET request (e.g. ~40 lines checking the
# local embedding model's cache on Hugging Face Hub) at INFO level, which
# drowns out our own logging on every run. WARNING keeps real problems visible.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    api_base: str
    model: str


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    local_model: str
    openai_model: str
    openai_api_key: str


def get_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    api_base_by_provider = {
        "groq": os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1"),
        "together": os.getenv("TOGETHER_API_BASE", "https://api.together.xyz/v1"),
        "openai": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    }
    api_key_by_provider = {
        "groq": os.getenv("GROQ_API_KEY", ""),
        "together": os.getenv("TOGETHER_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", ""),
    }
    if provider not in api_base_by_provider:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}; expected one of {list(api_base_by_provider)}"
        )
    return LLMConfig(
        provider=provider,
        api_key=api_key_by_provider[provider],
        api_base=api_base_by_provider[provider],
        # llama-3.3-70b-versatile was decommissioned on Groq; openai/gpt-oss-120b
        # is a current default. Check console.groq.com for the live model list.
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
    )


def get_eval_llm_config() -> LLMConfig:
    """Same provider/key/base-url as get_llm_config(), but the model can be
    overridden via EVAL_LLM_MODEL.

    Useful when the primary chat model's output format breaks RAGAS's
    JSON-based scoring prompts (observed with some gpt-oss responses) --
    point EVAL_LLM_MODEL at a different model (e.g. qwen/qwen3.8-27b) to use
    it only as the RAGAS judge, without changing the model that answers
    questions.
    """
    base = get_llm_config()
    eval_model = os.getenv("EVAL_LLM_MODEL", "").strip()
    if not eval_model:
        return base
    return LLMConfig(
        provider=base.provider,
        api_key=base.api_key,
        api_base=base.api_base,
        model=eval_model,
    )


def get_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider=os.getenv("EMBEDDING_PROVIDER", "local").lower(),
        local_model=os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        openai_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )


CHROMA_PERSIST_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "aml_kyc_knowledge_base")


# --- Database / vector backend ---------------------------------------------
# VECTOR_BACKEND selects where document chunks are stored and searched:
#   chroma   - local Chroma folder (default, zero setup)
#   pgvector - PostgreSQL with the pgvector extension (needs DATABASE_URL)
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma").lower()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # bge-small-en-v1.5 -> 384


def get_database_url() -> str:
    """DATABASE_URL for invoices / companies / logs. Falls back to a local SQLite file
    so everything works without PostgreSQL."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return f"sqlite:///{(PROJECT_ROOT / 'logs' / 'app.db').as_posix()}"
    # SQLAlchemy needs to be told to use the psycopg (v3) driver.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url
