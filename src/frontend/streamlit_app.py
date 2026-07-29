"""
Streamlit chat UI for the Crop Advisory ReAct Agent.

Calls the FastAPI backend (src/api/main.py) over HTTP — same pattern as
Project 2's Gradio frontend calling its own FastAPI backend locally.

Usage:
    # Terminal 1:
    uvicorn src.api.main:app --reload --port 8003

    # Terminal 2:
    streamlit run src/frontend/streamlit_app.py
    → http://localhost:8501
"""

import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8003/chat"

st.set_page_config(page_title="Crop Advisory Assistant", page_icon="🌽")
st.title("🌽 Crop Advisory Assistant")
st.caption(
    "Ask about corn yield forecasts or agronomic research. Combines a yield "
    "prediction model and a research literature search tool."
)

# Persist one session_id for this browser session, so conversation memory
# (Redis short-term, vector-store long-term) carries across turns within
# the same Streamlit session — not just within one HTTP request.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about corn yield or farming practices...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Thinking... With the local model, the FIRST question after starting "
            "the server can take several minutes (loading models); later questions "
            "are much faster."
        ):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question, "session_id": st.session_state.session_id},
                    # 180s wasn't enough for a real test: a multi-tool
                    # question with the LOCAL model can need to cold-start
                    # BOTH this project's Qwen3-4B AND Project 3's own
                    # embedder + generator (loaded when search_literature
                    # runs), plus multiple ReAct reasoning steps each
                    # needing their own local generation call. 600s gives
                    # real headroom for a cold start; subsequent requests
                    # are much faster since models stay cached in memory
                    # (singleton pattern — see react_agent.py).
                    timeout=600,
                )
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
            except requests.exceptions.ConnectionError:
                answer = (
                    "⚠ Could not reach the agent API. Make sure it's running: "
                    "`uvicorn src.api.main:app --reload --port 8003`"
                )
            except Exception as e:
                answer = f"⚠ Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    if st.button("Start new conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
