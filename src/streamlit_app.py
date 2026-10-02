"""Streamlit demo UI for the AML/KYC knowledge assistant.

Talks to the FastAPI backend (src/api.py) over HTTP, so the UI and the API
are independently deployable services (see docker-compose.yml, Step 9).

Run with:
    streamlit run src/streamlit_app.py
"""

from __future__ import annotations

import html
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

EXAMPLES = [
    ("What does beneficial ownership mean?", False),
    ("What CDD measures does FATF require?", False),
    ("How long can Azerbaijan freeze a suspicious transaction?", False),
    ("What's 450 times 37?", True),
    ("What is the latest FATF grey list update?", True),
]

CSS = """
<style>
.block-container { max-width: 880px; padding-top: 2rem; }
.hero {
    background: linear-gradient(135deg, #0f2a4a 0%, #1d4e89 100%);
    color: #fff; padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
}
.hero h1 { color: #fff; margin: 0 0 .3rem 0; font-size: 1.9rem; }
.hero p { color: #d6e4f5; margin: 0 0 .9rem 0; }
.stat { display: inline-block; background: rgba(255,255,255,.14); color: #fff;
        padding: .25rem .7rem; border-radius: 999px; font-size: .82rem; margin: 0 .4rem .3rem 0; }
.chip { display: inline-block; background: #eef3fa; color: #1d4e89; border: 1px solid #cfdcec;
        padding: .15rem .6rem; border-radius: 999px; font-size: .78rem; margin: 0 .35rem .35rem 0; }
.mode-rag, .mode-agent { display: inline-block; font-size: .72rem; font-weight: 600;
        padding: .1rem .5rem; border-radius: 6px; margin-bottom: .4rem; }
.mode-rag { background: #e6f4ea; color: #1e6b34; }
.mode-agent { background: #fff1db; color: #8a5300; }
</style>
"""


def ask_api(question: str, use_agent: bool, top_k: int) -> dict:
    endpoint = "/ask_agent" if use_agent else "/ask"
    response = requests.post(
        f"{API_URL}{endpoint}",
        json={"question": question, "top_k": top_k},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def render_sources(sources: list[dict]) -> None:
    seen, chips = set(), []
    for source in sources:
        key = (source["source"], source.get("page"))
        if key in seen:
            continue
        seen.add(key)
        page = f" · p.{source['page']}" if source.get("page") else ""
        chips.append(f"<span class='chip'>{html.escape(source['source'])}{page}</span>")
    if chips:
        st.markdown("**Sources**", help="Document passages the answer was based on.")
        st.markdown("".join(chips), unsafe_allow_html=True)


def render_assistant(entry: dict) -> None:
    data = entry["data"]
    mode_label = "Agent" if entry["agent"] else "Knowledge base"
    mode_class = "mode-agent" if entry["agent"] else "mode-rag"
    st.markdown(f"<span class='{mode_class}'>{mode_label} mode</span>", unsafe_allow_html=True)

    answer = data["answer"]
    if answer.strip().lower().startswith("i don't know"):
        st.info(answer, icon="🤷")
    else:
        st.markdown(answer)

    render_sources(data.get("sources") or [])

    if data.get("tool_calls"):
        with st.expander(f"🔧 Agent used {len(data['tool_calls'])} tool call(s)"):
            for call in data["tool_calls"]:
                st.markdown(f"**`{call['tool']}`**  \n`{call['input']}`")
                st.code(str(call["output"])[:600], language="text")

    st.caption(f"⏱ {data.get('latency_ms', 0) / 1000:.1f} s")


def run_question(question: str) -> None:
    use_agent, top_k = st.session_state.use_agent, st.session_state.top_k
    st.session_state.history.append({"role": "user", "text": question})
    try:
        with st.spinner("Searching the documents and writing an answer…"):
            data = ask_api(question, use_agent, top_k)
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
        st.session_state.history.append({"role": "error", "text": f"The API returned an error: {detail}"})
    except requests.exceptions.RequestException as exc:
        st.session_state.history.append({"role": "error", "text": f"Could not reach the API at {API_URL}: {exc}"})
    else:
        st.session_state.history.append({"role": "assistant", "data": data, "agent": use_agent})


st.set_page_config(page_title="AML/KYC Knowledge Assistant", page_icon="🏦", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Settings")
    st.toggle(
        "Agent mode",
        key="use_agent",
        value=False,
        help="Lets the assistant also use a calculator and live web search, "
        "not just the document knowledge base.",
    )
    st.slider(
        "Passages to retrieve", min_value=1, max_value=12, value=8, key="top_k",
        help="How many document passages are given to the model as context.",
    )
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.caption(
        "Answers come only from the indexed documents (FATF, Wolfsberg Group, Central Bank of "
        "Azerbaijan). If they don't contain the answer, the assistant says so. "
        "This is a demo, not legal or compliance advice."
    )

st.markdown(
    """
<div class="hero">
  <h1>🏦 AML/KYC Knowledge Assistant</h1>
  <p>Ask a compliance question and get an answer grounded in real regulatory documents, with page-level sources.</p>
  <span class="stat">16 documents</span><span class="stat">2,176 passages</span>
  <span class="stat">FATF · Wolfsberg · CBAR</span>
</div>
""",
    unsafe_allow_html=True,
)

def pick_example(text: str, needs_agent: bool) -> None:
    # Runs as a button callback, i.e. before the next script run, which is the
    # only point where a widget-bound value like `use_agent` may be changed.
    st.session_state.use_agent = needs_agent
    st.session_state.pending_question = text


if not st.session_state.history:
    st.markdown("**Try an example**")
    columns = st.columns(2)
    for i, (text, needs_agent) in enumerate(EXAMPLES):
        label = f"{text}" + ("  🤖" if needs_agent else "")
        columns[i % 2].button(
            label, key=f"ex{i}", use_container_width=True, on_click=pick_example, args=(text, needs_agent)
        )
    st.caption("🤖 = works best with Agent mode (the example switches it on for you).")
pending = st.session_state.pop("pending_question", None)

for entry in st.session_state.history:
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.markdown(entry["text"])
    elif entry["role"] == "assistant":
        with st.chat_message("assistant"):
            render_assistant(entry)
    else:
        st.error(entry["text"])

typed = st.chat_input("Ask about AML/KYC, e.g. “What is beneficial ownership?”")
question = typed or pending
if question:
    run_question(question)
    st.rerun()
