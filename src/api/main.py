"""
FastAPI serving layer for the Crop Advisory ReAct Agent.

Wraps ReactAgent behind a simple HTTP API — this is what the Streamlit
frontend (src/frontend/streamlit_app.py) calls, and what a live demo
deployment would expose.

NOTE ON AGENT INSTANCES: a new ReactAgent() is created per request rather
than cached globally, but this is cheap — the actual expensive part (LLM
model loading) is a CLASS-level singleton inside LocalQwenPipeline/
AnthropicLLM (see react_agent.py's `_pipe` class attribute), so only the
very first request in the process pays the model-loading cost. Session
memory (Redis/vector store) is keyed by session_id, not by ReactAgent
instance, so creating a fresh lightweight ReactAgent per request doesn't
lose any conversation continuity.

Usage:
    uvicorn src.api.main:app --reload --port 8003
    → http://localhost:8003/docs
"""

import logging
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.react_agent import ReactAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Crop Advisory Agent API",
    description="ReAct agent combining corn yield prediction and agronomic research search.",
)


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    iterations: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Runs one question through the agent. If session_id is omitted, a new
    one is generated and returned — the caller (e.g. the Streamlit
    frontend) should persist and resend it on subsequent calls to keep
    conversation memory continuous.
    """
    session_id = request.session_id or str(uuid.uuid4())

    agent = ReactAgent(session_id=session_id)
    result = agent.run(request.question, verbose=False)

    return ChatResponse(
        answer=result["answer"],
        session_id=session_id,
        iterations=result["iterations"],
    )
