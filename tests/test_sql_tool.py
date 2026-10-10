"""Tests for the agent's SQL and invoice tools, and for the routing scenarios file.

The SQL guard is tested hard on purpose: it is the part that stops an LLM-written
query from touching anything except two read-only views.
"""

import json
import os
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from src import db
from src.agent import TOOLS, _find_invoice_file, extract_invoice, sql_query
from src.sql_tool import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    SQLNotAllowed,
    format_rows,
    run_readonly_query,
    validate_sql,
)

ROOT = Path(__file__).resolve().parent.parent
TEST_PG_URL = os.getenv("TEST_DATABASE_URL", "postgresql+psycopg://assistant:assistant@localhost:5432/assistant")


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'sql.db').as_posix()}")
    db.init_db(engine)
    db.seed_companies(engine)
    for number, (currency, amount) in enumerate([("AZN", "100.50"), ("AZN", "200.00"), ("EUR", "50.00")], start=1):
        db.save_invoice(
            {"file_name": f"{number}.pdf", "company_id": number, "company_name": f"Co {number}", "voen": "1234567890",
             "invoice_number": f"N-{number}", "issue_date": date(2026, 1, number), "total_amount": Decimal(amount),
             "currency": currency, "needs_human_review": number == 3},
            engine,
        )
    return engine


# --- validation -----------------------------------------------------------------

def test_plain_select_on_an_allowed_view_passes_and_gets_a_default_limit():
    sql = validate_sql("SELECT id, total_amount FROM v_invoices WHERE currency = 'AZN'")
    assert sql.upper().endswith(f"LIMIT {DEFAULT_LIMIT}")


def test_an_explicit_small_limit_is_kept():
    assert validate_sql("SELECT * FROM v_companies LIMIT 5").upper().endswith("LIMIT 5")


@pytest.mark.parametrize(
    "bad_sql",
    [
        "DROP TABLE invoices",
        "DELETE FROM v_invoices",
        "UPDATE v_invoices SET total_amount = 0",
        "INSERT INTO v_companies VALUES (1, 'x', '1')",
        "SELECT * FROM v_invoices; DROP TABLE invoices",
        "SELECT * FROM invoices",                      # the base table, not the view
        "SELECT * FROM companies",
        "SELECT * FROM pg_user",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM pg_catalog.pg_shadow",
        "SELECT * FROM v_invoices JOIN invoices ON 1=1",
        "SELECT * FROM v_invoices, companies",
        "SELECT * FROM v_invoices UNION SELECT * FROM v_companies",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "SELECT pg_sleep(100) FROM v_invoices",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT set_config('x', 'y', false)",
        "SELECT * INTO newtable FROM v_invoices",
        "SELECT * FROM v_invoices FOR UPDATE",
        "SELECT * FROM v_invoices LIMIT 100000",
        "SELECT * FROM v_invoices LIMIT (SELECT 5)",
        "not sql at all (((",
        "",
    ],
)
def test_dangerous_or_out_of_scope_queries_are_rejected(bad_sql):
    with pytest.raises(SQLNotAllowed):
        validate_sql(bad_sql)


def test_limit_at_the_maximum_is_allowed():
    validate_sql(f"SELECT * FROM v_invoices LIMIT {MAX_LIMIT}")


# --- execution (SQLite) -----------------------------------------------------------

def test_aggregate_query_returns_rows(engine):
    rows = run_readonly_query("SELECT currency, SUM(total_amount) AS total FROM v_invoices GROUP BY currency ORDER BY currency", engine)
    assert [r["currency"] for r in rows] == ["AZN", "EUR"]
    assert float(rows[0]["total"]) == pytest.approx(300.5)


def test_the_view_does_not_expose_the_voen_of_invoices(engine):
    with pytest.raises(Exception):  # noqa: B017 - "no such column"
        run_readonly_query("SELECT voen FROM v_invoices", engine)


def test_writes_are_impossible_even_if_validation_were_bypassed(engine):
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA query_only = ON")
        with pytest.raises(Exception):  # noqa: B017 - "attempt to write a readonly database"
            conn.execute(text("DELETE FROM invoices"))


def test_format_rows_handles_empty_and_long_results():
    assert "no rows" in format_rows([])
    assert "truncated" in format_rows([{"x": "y" * 5000}])


# --- the tools as the agent sees them -----------------------------------------------

def test_sql_query_tool_reports_rejections_as_text_not_exceptions():
    assert sql_query.invoke({"query": "DROP TABLE invoices"}).startswith("Query rejected")


def test_sql_query_tool_runs_a_query(engine, monkeypatch):
    import src.sql_tool as sql_module

    monkeypatch.setattr(sql_module, "_readonly_engine", lambda: engine)
    assert "AZN" in sql_query.invoke({"query": "SELECT DISTINCT currency FROM v_invoices"})


def test_extract_invoice_rejects_path_traversal_and_lists_files():
    assert _find_invoice_file("../../.env") is None
    assert _find_invoice_file("inv_01_text.pdf") is not None
    message = extract_invoice.invoke({"file_name": "../../etc/passwd"})
    assert "not found" in message and "inv_01_text.pdf" in message


def test_extract_invoice_tool_returns_json_with_company_match(monkeypatch):
    import src.agent as agent_module
    from src.invoices.extract import ExtractionResult
    from src.invoices.schema import InvoiceData

    label = json.loads((ROOT / "data" / "invoices" / "labels.json").read_text(encoding="utf-8"))[0]
    invoice = InvoiceData.model_validate({k: label[k] for k in
                                          ["company_name", "voen", "invoice_number", "issue_date", "total_amount", "currency"]})
    monkeypatch.setattr(agent_module, "extract_invoice_file", lambda path: ExtractionResult(invoice=invoice, attempts=1))
    payload = json.loads(extract_invoice.invoke({"file_name": label["file"]}))
    assert payload["invoice"]["voen"] == label["voen"]
    assert payload["company_match"]["method"] == "voen" and payload["company_match"]["company_id"] == label["reference_company_id"]


def test_agent_has_all_five_tools():
    assert {t.name for t in TOOLS} == {
        "search_knowledge_base", "calculator", "internet_search", "sql_query", "extract_invoice"
    }


# --- routing scenarios file -----------------------------------------------------------

def test_scenarios_reference_only_existing_tools_and_unique_ids():
    scenarios = json.loads((ROOT / "data" / "agent_test_scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    names = {t.name for t in TOOLS}
    ids = [s["id"] for s in scenarios]
    assert len(ids) == len(set(ids)) >= 16
    for scenario in scenarios:
        assert scenario["expected_route"] in {"tool", "rag"}
        assert (scenario["expected_tool"] in names) == (scenario["expected_route"] == "tool")
    assert {"sql_query", "extract_invoice"} <= {s["expected_tool"] for s in scenarios}


def test_routing_pass_rules():
    from scripts.check_agent_routing import passed

    assert passed({"expected_route": "tool", "expected_tool": "sql_query"}, {"sql_query", "calculator"})
    assert not passed({"expected_route": "tool", "expected_tool": "sql_query"}, {"calculator"})
    assert passed({"expected_route": "rag", "expected_tool": None}, {"search_knowledge_base"})
    assert not passed({"expected_route": "rag", "expected_tool": None}, {"sql_query"})


# --- real PostgreSQL: the read-only role ----------------------------------------------------

@pytest.mark.postgres
def test_postgres_readonly_role_can_only_read_the_views():
    try:
        admin = create_engine(TEST_PG_URL)
        db.init_db(admin)
        db.seed_companies(admin)
    except Exception:
        pytest.skip("PostgreSQL is not reachable")
    readonly = create_engine(TEST_PG_URL.replace("assistant:assistant@", "assistant_ro:assistant_ro@"))
    rows = run_readonly_query("SELECT id, name FROM v_companies ORDER BY id LIMIT 3", readonly)
    assert len(rows) == 3
    with pytest.raises(Exception):  # noqa: B017 - permission denied on the base table
        with readonly.connect() as conn:
            conn.execute(text("SELECT * FROM companies"))
    with pytest.raises(Exception):  # noqa: B017 - read-only transaction
        with readonly.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text("DELETE FROM v_companies"))


def test_a_query_does_not_leave_the_connection_read_only(engine):
    """Regression: PRAGMA query_only used to stay on after a query, so later log writes failed."""
    run_readonly_query("SELECT * FROM v_companies LIMIT 1", engine)
    db.log_request_row("q", "a", [], "rag", 1.0, engine)  # would raise "readonly database" before the fix
    assert db.count_rows(db.request_logs, engine) == 1
