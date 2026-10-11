"""MCP server tests. They run only when the optional `mcp` package is installed (requirements-mcp.txt)."""

import asyncio

import pytest

pytest.importorskip("mcp.server.mcpserver")

from src import mcp_server  # noqa: E402


def call(name, arguments):
    result = asyncio.run(mcp_server.mcp.call_tool(name, arguments))
    return result.content[0].text


def test_server_exposes_the_five_tools():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {t.name for t in tools} == {
        "search_knowledge_base", "sql_query", "extract_invoice", "translate_text", "summarize_text"
    }
    assert all(t.description for t in tools)


def test_sql_tool_keeps_the_read_only_guard():
    assert call("sql_query", {"query": "DROP TABLE invoices"}).startswith("Query rejected")


def test_invoice_tool_keeps_the_path_traversal_check():
    text = call("extract_invoice", {"file_name": "../../etc/passwd"})
    assert "not found" in text


def test_translate_tool_calls_the_text_tools(monkeypatch):
    monkeypatch.setattr(mcp_server.text_tools, "translate", lambda text, target: f"[{target}] {text}")
    assert call("translate_text", {"text": "hi", "target_language": "az"}) == "[az] hi"
