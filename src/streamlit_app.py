"""Streamlit demo UI for the AML/KYC knowledge assistant.

Talks to the FastAPI backend (src/api.py) over HTTP, so the UI and the API
are independently deployable services (see docker-compose.yml, Step 9).

Run with:
    streamlit run src/streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AML/KYC Knowledge Assistant", page_icon="🏦")
st.title("🏦 AML/KYC Knowledge Assistant")
st.caption(
    "Ask a question about AML/KYC compliance for a neobank/fintech. "
    "Answers are grounded in FATF, Wolfsberg Group, and CBAR (Azerbaijan) source documents."
)

use_agent = st.sidebar.toggle(
    "Use agentic mode (tool-calling)",
    value=False,
    help="When on, the assistant can also use a calculator or web search, "
    "not just the document knowledge base.",
)
top_k = st.sidebar.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=4)

with st.sidebar:
    st.markdown("---")
    st.markdown(
        "**Example questions**\n"
        "- What does beneficial ownership mean?\n"
        "- What CDD measures does FATF require?\n"
        "- How long can Azerbaijan freeze a suspicious transaction?\n"
        "- What's 450 times 37? *(agentic mode)*\n"
        "- What's today's USD/AZN exchange rate? *(agentic mode)*"
    )

question = st.text_input("Your question", placeholder="e.g. What is beneficial ownership?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    endpoint = "/ask_agent" if use_agent else "/ask"
    with st.spinner("Retrieving context and generating an answer..."):
        try:
            response = requests.post(
                f"{API_URL}{endpoint}",
                json={"question": question, "top_k": top_k},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the API at {API_URL}: {exc}")
        else:
            st.markdown("### Answer")
            st.write(data["answer"])

            if data.get("sources"):
                st.markdown("### Sources")
                for source in data["sources"]:
                    page = source.get("page")
                    label = f"{source['source']}" + (f", p. {page}" if page else "")
                    st.markdown(f"- `{label}`")

            if data.get("tool_calls"):
                st.markdown("### Tool calls made by the agent")
                for call in data["tool_calls"]:
                    st.code(f"{call['tool']}({call['input']}) -> {call['output']}", language="text")

            st.caption(f"Latency: {data.get('latency_ms', 0):.0f} ms")
elif ask_clicked:
    st.warning("Please enter a question first.")
