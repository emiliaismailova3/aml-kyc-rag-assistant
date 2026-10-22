"""When the agent loops until its step limit, the base RAG pipeline answers instead."""

from langgraph.errors import GraphRecursionError

import src.agent as agent_module
import src.rag as rag_module


class LoopingExecutor:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload, config):
        self.calls += 1
        raise GraphRecursionError("Recursion limit of 10 reached")


class FakeRAG:
    def __init__(self, *args, **kwargs):
        pass

    def answer(self, question, k=None):
        return {"answer": "Owners are identified [1].", "sources": [{"ref": 1, "source": "x.pdf", "page": 2}],
                "contexts": ["[1] (Source: x.pdf, p. 2)\nOwners ..."]}


def make_agent():
    pipeline = agent_module.AgentPipeline.__new__(agent_module.AgentPipeline)  # skip building a real LLM
    pipeline.executor, pipeline._rag_pipeline = LoopingExecutor(), None
    return pipeline


def test_step_limit_falls_back_to_rag(monkeypatch):
    monkeypatch.setattr(rag_module, "RAGPipeline", FakeRAG)
    pipeline = make_agent()
    result = pipeline.answer("What CDD measures does FATF require?")
    assert result["fallback"] == "rag_after_step_limit" and "error" not in result
    assert result["answer"] == "Owners are identified [1]." and result["sources"][0]["source"] == "x.pdf"
    assert result["tool_calls"][0]["tool"] == "rag_fallback"
    assert pipeline.executor.calls == 1  # a step-limit loop is not retried


def test_fallback_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("AGENT_RAG_FALLBACK", "0")
    monkeypatch.setattr(rag_module, "RAGPipeline", FakeRAG)
    result = make_agent().answer("q")
    assert result["error"] == "GraphRecursionError" and "fallback" not in result


def test_prompt_caps_knowledge_base_searches():
    assert "search_knowledge_base: call it AT MOST 3 times" in agent_module._build_system_prompt()


def test_agent_has_hard_tool_call_limits():
    from langchain.agents.middleware import ToolCallLimitMiddleware

    assert agent_module.MAX_KB_SEARCHES == 3 and agent_module.MAX_WEB_SEARCHES == 2
    assert ToolCallLimitMiddleware(tool_name="search_knowledge_base", run_limit=3)  # constructible in this LangChain version


def test_fallback_that_cannot_reach_the_llm_returns_an_error_not_an_exception(monkeypatch):
    from src.llm_client import LLMUnavailable

    class DownRAG(FakeRAG):
        def answer(self, question, k=None):
            raise LLMUnavailable("all providers failed")

    monkeypatch.setattr(rag_module, "RAGPipeline", DownRAG)
    result = make_agent().answer("q")
    assert result["error"] == "LLMUnavailable"


def test_a_blocked_search_does_not_hide_the_real_sources():
    from langchain_core.messages import AIMessage, ToolMessage

    real = "[1] (Source: fatf_recommendations_2025.pdf, p. 15)\nCDD measures ..."

    class Executor:
        def invoke(self, payload, config):
            call = lambda i: {"name": "search_knowledge_base", "args": {"query": "q"}, "id": f"c{i}"}  # noqa: E731
            return {"messages": [
                AIMessage(content="", tool_calls=[call(1)]), ToolMessage(content=real, tool_call_id="c1"),
                AIMessage(content="", tool_calls=[call(2)]),
                ToolMessage(content="Tool call limit exceeded. Do not call search_knowledge_base again.", tool_call_id="c2"),
                AIMessage(content="Institutions must identify the customer [1]."),
            ]}

    pipeline = make_agent()
    pipeline.executor = Executor()
    result = pipeline.answer("q")
    assert result["sources"] and result["sources"][0]["source"] == "fatf_recommendations_2025.pdf"
