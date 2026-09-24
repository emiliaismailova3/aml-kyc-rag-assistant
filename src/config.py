"""Central place for reading configuration from environment variables (.env).

Keeping this in one module means every other module (vectorstore, rag, agent,
api) reads settings the same way and it's obvious what needs to be set in
.env for the project to run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


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
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
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
