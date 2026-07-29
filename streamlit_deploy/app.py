"""
Streamlit Cloud deployment entry point — single app, no separate FastAPI
process (Streamlit Cloud can't run two services the way local dev does;
same lesson learned deploying Project 2 to Hugging Face Spaces). Calls
the agent DIRECTLY in-process.

To deploy: see DEPLOY.md in this folder.
"""

import uuid

import streamlit as st

from agent import DeployedAgent

st.set_page_config(page_title="Crop Advisory Assistant", page_icon="🌽")
st.title("🌽 Crop Advisory Assistant")
st.caption(
    "Ask about corn yield forecasts or agronomic research. Combines a yield "
    "prediction model and a research literature search tool, powered by "
    "Claude Sonnet 4.5."
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = DeployedAgent(session_id=st.session_state.session_id)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about corn yield or farming practices...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.agent.run(question)
                answer = result["answer"]
            except Exception as e:
                answer = f"⚠ Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    st.caption("Powered by Claude Sonnet 4.5")
    if st.button("Start new conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.agent = DeployedAgent(session_id=st.session_state.session_id)
        st.rerun()

    st.divider()
    st.caption(
        "Full project source and evaluation results: "
        "[GitHub](https://github.com/linz21/crop_advisory_react_agent)"
    )
