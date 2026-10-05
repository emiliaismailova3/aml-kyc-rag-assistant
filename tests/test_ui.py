"""Smoke tests for the Streamlit UI (src/streamlit_app.py) using Streamlit's
built-in AppTest runner, with the backend API call mocked -- no server, LLM key
or network needed."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "src" / "streamlit_app.py")


def _fake_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _markdown_text(app):
    return " ".join(m.value for m in app.markdown)


def test_answer_shows_deduplicated_source_chips():
    payload = {
        "answer": "Beneficial ownership means ultimate control [1].",
        "sources": [
            {"source": "wolfsberg_faqs_beneficial_ownership.pdf", "page": 2},
            {"source": "wolfsberg_faqs_beneficial_ownership.pdf", "page": 2},
            {"source": "fatf_recommendations_2025.pdf", "page": 123},
        ],
        "tool_calls": [],
        "latency_ms": 1500,
    }
    with patch("requests.post", return_value=_fake_response(payload)) as post:
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.chat_input[0].set_value("What is beneficial ownership?").run()

    assert not app.exception
    text = _markdown_text(app)
    assert "Beneficial ownership means ultimate control" in text
    assert text.count("wolfsberg_faqs_beneficial_ownership.pdf") == 1, "duplicate sources should collapse to one chip"
    assert "fatf_recommendations_2025.pdf · p.123" in text
    assert post.call_args.args[0].endswith("/ask")


def test_refusal_is_shown_as_info_box_without_sources():
    payload = {"answer": "I don't know based on the available documents.", "sources": [], "tool_calls": [], "latency_ms": 900}
    with patch("requests.post", return_value=_fake_response(payload)):
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.chat_input[0].set_value("How do I cook plov?").run()

    assert not app.exception
    assert any("I don't know" in i.value for i in app.info)
    assert "Sources" not in _markdown_text(app)


def test_api_unreachable_shows_error_instead_of_crashing():
    import requests

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("down")):
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.chat_input[0].set_value("anything").run()

    assert not app.exception
    assert any("Could not reach the API" in e.value for e in app.error)


def test_passages_slider_is_disabled_in_agent_mode():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.slider[0].disabled
    app.toggle[0].set_value(True).run()
    assert app.slider[0].disabled


def test_source_chips_show_citation_numbers():
    payload = {
        "answer": "Control means ultimate ownership [5][3].",
        "sources": [
            {"ref": 5, "source": "wolfsberg_faqs_beneficial_ownership.pdf", "page": 4},
            {"ref": 3, "source": "fatf_guidance_beneficial_ownership_legal_persons.pdf", "page": 3},
        ],
        "tool_calls": [],
        "latency_ms": 1200,
    }
    with patch("requests.post", return_value=_fake_response(payload)):
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.chat_input[0].set_value("What is beneficial ownership?").run()

    assert not app.exception
    text = _markdown_text(app)
    assert "<b>[5]</b> wolfsberg_faqs_beneficial_ownership.pdf · p.4" in text
    assert "<b>[3]</b> fatf_guidance_beneficial_ownership_legal_persons.pdf · p.3" in text
