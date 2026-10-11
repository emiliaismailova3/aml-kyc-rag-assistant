"""MCP server: the assistant's tools, exposed so that any MCP client (Claude Desktop, an IDE, another
agent) can call them. It reuses the exact same functions the built-in agent uses, so the SQL guard,
the path-traversal check and the LLM reliability layer apply here too.

Tools: search_knowledge_base, sql_query (read-only, two views), extract_invoice, translate_text, summarize_text.

Run (stdio transport, which is what desktop MCP clients launch):
    python -m src.mcp_server

Client configuration example (Claude Desktop, claude_desktop_config.json):
    {"mcpServers": {"aml-kyc-assistant": {"command": "python", "args": ["-m", "src.mcp_server"],
                                          "cwd": "<path to this repository>"}}}

Needs the optional dependency from requirements-mcp.txt (mcp 2.x).
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from src import agent, text_tools

mcp = MCPServer(
    "aml-kyc-assistant",
    instructions=(
        "AML/KYC knowledge assistant. Use search_knowledge_base for compliance questions (it cites sources), "
        "sql_query for facts about invoices and companies, extract_invoice to read an invoice file."
    ),
)


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Search the AML/KYC documents (FATF, Wolfsberg, CBAR) and return the most relevant excerpts with sources."""
    return agent.search_knowledge_base.invoke({"query": query})


@mcp.tool()
def sql_query(query: str) -> str:
    """Run one read-only SELECT on the views v_invoices and v_companies. Anything else is rejected."""
    return agent.sql_query.invoke({"query": query})


@mcp.tool()
def extract_invoice(file_name: str) -> str:
    """Extract validated structured data from an invoice in data/invoices and match the company to the reference list."""
    return agent.extract_invoice.invoke({"file_name": file_name})


@mcp.tool()
def translate_text(text: str, target_language: str = "en") -> str:
    """Translate text into en, az, ru or tr. Names, numbers and identifiers are kept as written."""
    return text_tools.translate(text, target_language)


@mcp.tool()
def summarize_text(text: str, language: str = "en", max_sentences: int = 5) -> str:
    """Summarize text in en, az, ru or tr using only facts from the text."""
    return text_tools.summarize(text, language, max_sentences)


if __name__ == "__main__":
    mcp.run("stdio")
