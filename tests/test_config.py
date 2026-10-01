"""Tests for src/config.py, in particular the EVAL_LLM_MODEL override used
to let the RAGAS judge model differ from the model that answers questions
(src/evaluate.py), without needing any real API key or network access."""

from src.config import get_eval_llm_config, get_llm_config


def test_eval_llm_config_defaults_to_main_model(monkeypatch):
    monkeypatch.delenv("EVAL_LLM_MODEL", raising=False)
    main_config = get_llm_config()
    eval_config = get_eval_llm_config()
    assert eval_config.model == main_config.model
    assert eval_config.provider == main_config.provider
    assert eval_config.api_key == main_config.api_key


def test_eval_llm_config_override(monkeypatch):
    monkeypatch.setenv("EVAL_LLM_MODEL", "qwen/qwen3.8-27b")
    eval_config = get_eval_llm_config()
    main_config = get_llm_config()
    assert eval_config.model == "qwen/qwen3.8-27b"
    # Everything else (provider, key, base URL) stays the same as the main config.
    assert eval_config.provider == main_config.provider
    assert eval_config.api_key == main_config.api_key
    assert eval_config.api_base == main_config.api_base
