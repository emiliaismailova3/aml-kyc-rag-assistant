"""Tests for the resumable RAGAS scoring logic in src/evaluate.py.

The real judge LLM is replaced by a fake scorer (src.evaluate._build_scorer is
monkeypatched), so these check the caching / early-stop / aggregation behaviour
without any API key, network access, or token cost.
"""

import json

import src.evaluate as ev

FULL = {"faithfulness": 1.0, "answer_relevancy": 0.5, "answer_correctness": 0.25}
EMPTY = {"faithfulness": None, "answer_relevancy": None, "answer_correctness": None}


def _samples(n):
    return [{"id": f"q{i}", "question": "?", "answer": "a", "contexts": ["c"], "ground_truth": "g"} for i in range(n)]


def test_aggregate_skips_missing_values():
    rows = [
        {"faithfulness": 1.0, "answer_relevancy": None, "answer_correctness": 0.5},
        {"faithfulness": 0.0, "answer_relevancy": 0.4, "answer_correctness": 0.5},
    ]
    agg = ev._aggregate(rows)
    assert agg == {"faithfulness": 0.5, "answer_relevancy": 0.4, "answer_correctness": 0.5}


def test_only_fully_scored_questions_are_cached_and_resumed(monkeypatch, tmp_path):
    cache = tmp_path / "cache.json"
    calls = []

    def fake_scorer():
        def score_one(sample):
            calls.append(sample["id"])
            return dict(FULL) if sample["id"] != "q1" else {**FULL, "faithfulness": None}
        return score_one

    monkeypatch.setattr(ev, "_build_scorer", fake_scorer)
    complete, incomplete = ev._score_samples_with_resume(_samples(3), cache)
    assert set(complete) == {"q0", "q2"}
    assert set(incomplete) == {"q1"}
    assert {r["id"] for r in json.loads(cache.read_text())} == {"q0", "q2"}

    calls.clear()
    complete, _ = ev._score_samples_with_resume(_samples(3), cache)
    assert calls == ["q1"], "only the incomplete question should be re-scored on resume"


def test_stops_early_after_consecutive_empty_scores(monkeypatch, tmp_path):
    calls = []

    def fake_scorer():
        def score_one(sample):
            calls.append(sample["id"])
            return dict(EMPTY)
        return score_one

    monkeypatch.setattr(ev, "_build_scorer", fake_scorer)
    complete, incomplete = ev._score_samples_with_resume(_samples(10), tmp_path / "c.json")
    assert complete == {}
    assert calls == ["q0", "q1"], "should give up after 2 questions with no scores, not grind through all 10"


def test_failed_agent_answers_are_not_cached(monkeypatch, tmp_path):
    import pytest

    import src.agent as agent_module

    class FakeAgent:
        def answer(self, question):
            return {"question": question, "answer": "I couldn't complete this request", "tool_calls": [], "error": "RateLimitError"}

    monkeypatch.setattr(agent_module, "AgentPipeline", FakeAgent)
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)
    questions = [{"id": "q1", "question": "?", "ground_truth": "g"}]
    with pytest.raises(RuntimeError, match="not caching"):
        ev._collect_agent_samples(questions)
    assert not (tmp_path / "agent_samples_cache.json").exists() or json.loads((tmp_path / "agent_samples_cache.json").read_text()) == []


def test_agent_step_limit_is_recorded_as_a_real_result(monkeypatch, tmp_path):
    import src.agent as agent_module

    class LoopingAgent:
        def answer(self, question):
            return {"question": question, "answer": "I couldn't complete this request", "tool_calls": [], "error": "GraphRecursionError"}

    monkeypatch.setattr(agent_module, "AgentPipeline", LoopingAgent)
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ev, "INTER_QUESTION_DELAY_SECONDS", 0)
    samples = ev._collect_agent_samples([{"id": "q1", "question": "?", "ground_truth": "g"}])
    assert [s["id"] for s in samples] == ["q1"]
