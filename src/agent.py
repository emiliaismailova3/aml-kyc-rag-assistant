"""Agentic tool-calling layer for the AML/KYC knowledge assistant (Step 7).

Unlike src.rag.RAGPipeline (which always retrieves from the knowledge base
and stuffs the result into the prompt), this agent decides for itself, per
question, whether to:
  - answer directly,
  - call `search_knowledge_base` (the same Chroma retriever as the base RAG
    pipeline, exposed as a tool),
  - call `calculator` for arithmetic/percentage questions,
  - call `internet_search` (ddgs/DuckDuckGo, free, no API key) for live or
    out-of-corpus information, or
  - admit it doesn't know.

See data/agent_test_scenarios.json for the 10 scenarios (5 tool-requiring,
5 knowledge-base-only) used to validate this routing behaviour.

Usage:
    python -m src.agent "What's 450 times 37?"
"""

from __future__ import annotations

import argparse
import ast
import logging
import operator
from datetime import date
from typing import TypedDict

from ddgs import DDGS
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.rag import clean_citations, format_context, get_llm
from src.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)

# Caps the agent's reasoning loop (model turn + tool turn = 1 "super-step"
# each in LangGraph) so a model that gets stuck re-searching (observed: 7
# internet_search calls in a row for one question) can't run away burning
# rate-limited API calls. ~10 allows e.g. 2 searches + a knowledge-base
# lookup + a calculator call with room to spare, while still failing fast.
AGENT_RECURSION_LIMIT = 10

# Non-rate-limit tool errors (e.g. Groq's 400 tool_use_failed when a model
# passes malformed tool arguments) are retried once before giving up, since
# they're usually a one-off glitch in how the model formatted a single call
# rather than a persistent problem.
AGENT_MAX_ATTEMPTS = 2

# --- Calculator tool -------------------------------------------------------
# A small hand-rolled safe evaluator (ast-based, whitelist of operators) is
# used instead of Python's eval() so the LLM can never inject arbitrary code
# through a crafted "expression" argument -- only arithmetic is possible.

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression component: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression: +, -, *, /, %, ** and parentheses.
    Use this for ANY numeric calculation, percentage, or comparison (e.g. checking
    whether a shareholding percentage clears a threshold, or multiplying a fee by a
    count) instead of computing it yourself. Example input: "450 * 37" or "12.5 - 12".
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any parse/eval
        # failure should be reported back to the agent as a tool error, not crash it.
        return f"Error evaluating expression {expression!r}: {exc}"


# --- Web search tool ---------------------------------------------------------
# Uses the `ddgs` package (duckduckgo_search was renamed upstream; the old
# package now just warns and re-exports this one). region="us-en" avoids
# locale-dependent results (observed: a non-English local-business login page
# for a generic query) when the host machine's locale isn't English.


@tool
def internet_search(query: str) -> str:
    """Search the live web for current information that is NOT in the static
    AML/KYC knowledge base -- e.g. today's exchange rates, the current FATF
    grey list, recent news, or general knowledge unrelated to the compliance
    corpus. Returns the top few result titles, snippets, and URLs.

    Args:
        query: a plain-text search query string, e.g. "FATF grey list 2026".
            Pass ONLY this single string argument -- do not pass structured
            or browser-tool-style arguments such as {"cursor": ..., "id": ...}.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="us-en", max_results=5))
    except Exception as exc:  # noqa: BLE001 - network/search errors reported to the agent
        return f"Web search failed ({type(exc).__name__}): {exc}. Try a different query or answer without it."
    if not results:
        return f"No web results found for {query!r}. Try a different, more specific query."
    return "\n".join(
        f"- {r.get('title')}: {r.get('body')} ({r.get('href')})" for r in results
    )


# --- Knowledge base retrieval tool -------------------------------------------


@tool
def search_knowledge_base(query: str) -> str:
    """Search the AML/KYC document knowledge base (FATF Recommendations, Wolfsberg
    Group guidance, and Central Bank of Azerbaijan / Azerbaijani AML law) for
    passages relevant to the query. Use this for any question about AML/KYC policy,
    regulation, due diligence, or compliance concepts. Returns the top matching
    excerpts, each labeled with its source document and page number.
    """
    store = load_vectorstore()
    chunks = store.similarity_search(query, k=4)
    if not chunks:
        return "No relevant documents found in the knowledge base."
    return format_context(chunks)


TOOLS = [search_knowledge_base, calculator, internet_search]


def _build_system_prompt() -> str:
    """Built fresh per AgentPipeline construction (not a module-level
    constant) so a long-running process always tells the model today's real
    date -- without it, the model has no way to know "current" and searched
    for e.g. "FATF grey list 2024" when asked about today's status."""
    today = date.today().isoformat()
    return f"""You are an AML/KYC compliance assistant for a neobank/fintech.
Today's date is {today}.

You have three tools:
- search_knowledge_base: for AML/KYC policy, regulation, and compliance questions.
- calculator: for arithmetic, percentages, or numeric comparisons.
- internet_search: for live or current information not in the static knowledge base
  (exchange rates, current regulatory list membership, general knowledge, etc.).
  Use today's date above to judge what "current" or "latest" means -- do not
  guess or default to a stale year.

For every question, decide whether you can answer directly, or need one or more
tools. Prefer search_knowledge_base for anything about AML/KYC obligations,
definitions, or thresholds -- do not answer those from memory, since your own
knowledge may be outdated or wrong for a specific jurisdiction. Use calculator
for any arithmetic rather than computing it yourself. Use internet_search only
for information that is live, time-sensitive, or clearly outside the AML/KYC
corpus -- and call it AT MOST 2 times for a single question. If two searches
with different queries haven't found it, stop searching and either answer with
what you have (saying what's uncertain) or say you don't know; do not keep
repeating or rephrasing the same search.

If, after using the appropriate tool(s), you still don't have enough information
to answer confidently, say exactly: "I don't know based on the available
information." Do not fabricate AML/KYC obligations, thresholds, or deadlines.

When you rely on the knowledge base, cite the numbered context chunk(s) with
plain bracketed numbers matching the [1], [2], ... labels returned by
search_knowledge_base (e.g. "beneficial owners must be identified [1]"). Do not
use any other citation format. When you use the calculator or internet_search,
make clear in your answer that you did so.
"""


class ToolCallRecord(TypedDict):
    tool: str
    input: dict | str
    output: str


class AgentResult(TypedDict):
    question: str
    answer: str
    tool_calls: list[ToolCallRecord]


def build_agent_executor():
    """Construct the LangChain tool-calling agent (requires an LLM API key).

    Uses langchain.agents.create_agent (LangChain >=1.0), which compiles a
    LangGraph graph that loops between the model and the tools until the
    model stops requesting tool calls.
    """
    llm = get_llm()
    return create_agent(model=llm, tools=TOOLS, system_prompt=_build_system_prompt())


class AgentPipeline:
    """Thin wrapper around the compiled agent graph exposing a simple
    .answer() API that mirrors src.rag.RAGPipeline.answer(), so src/api.py
    and src/evaluate.py can treat both pipelines the same way."""

    def __init__(self):
        self.executor = build_agent_executor()

    def answer(self, question: str) -> AgentResult:
        # Rate-limit (429) retries happen automatically inside the LLM client
        # itself (see src.rag.LLM_MAX_RETRIES). This loop instead covers
        # one-off tool-calling glitches (e.g. Groq's 400 tool_use_failed when
        # a model sends malformed tool arguments) and a hard recursion cap
        # (AGENT_RECURSION_LIMIT), so one bad question can't crash a whole
        # batch evaluation run or burn the rate limit on a runaway loop.
        result = None
        last_exc: Exception | None = None
        for attempt in range(1, AGENT_MAX_ATTEMPTS + 1):
            try:
                result = self.executor.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config={"recursion_limit": AGENT_RECURSION_LIMIT},
                )
                break
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any
                # failure here (bad tool call, recursion limit, transport
                # error) should degrade to a graceful answer, not crash.
                last_exc = exc
                logger.warning(
                    "Agent invocation failed (attempt %d/%d) for %r: %s",
                    attempt,
                    AGENT_MAX_ATTEMPTS,
                    question,
                    exc,
                )

        if result is None:
            return {
                "question": question,
                "answer": (
                    "I couldn't complete this request due to a tool-calling error "
                    f"({type(last_exc).__name__ if last_exc else 'unknown'}). "
                    "Please try rephrasing the question."
                ),
                "tool_calls": [],
            }

        messages = result["messages"]

        # Match each AIMessage's tool_calls to the ToolMessage that carries
        # that call's result, by tool_call_id, to build a transparent log of
        # what the agent did before producing its final answer.
        tool_messages_by_id = {
            m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)
        }
        tool_calls: list[ToolCallRecord] = []
        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    observation = tool_messages_by_id.get(call["id"])
                    tool_calls.append(
                        {
                            "tool": call["name"],
                            "input": call["args"],
                            "output": str(observation.content)[:1000] if observation else "",
                        }
                    )

        final_answer = clean_citations(messages[-1].content)
        return {"question": question, "answer": final_answer, "tool_calls": tool_calls}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="A question for the agent to answer")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pipeline = AgentPipeline()
    result = pipeline.answer(args.question)
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    if result["tool_calls"]:
        print("Tool calls:")
        for call in result["tool_calls"]:
            print(f"  - {call['tool']}({call['input']}) -> {call['output'][:200]}")
    else:
        print("(No tools were called -- answered directly.)")


if __name__ == "__main__":
    _main()
