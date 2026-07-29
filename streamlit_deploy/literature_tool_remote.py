"""
Literature search tool for the Streamlit Cloud deployment — calls Project
2's ALREADY-DEPLOYED Hugging Face Space directly via gradio_client, rather
than importing Project 2's code in-process (which would require bundling
its entire RAG stack — models, Chroma index, embeddings — into THIS
deployment too). Project 2 is already live and working; reusing it avoids
duplicating that infrastructure.

API CONFIRMED against Project 2's actual "Use via API" page:
    api_name: "/ask_question" (a duplicate "/ask_question_1" also exists,
    from the same function being bound to two UI events in Project 2's
    app — either works identically; using the first).
    Params: question (str), history (list, default [])
    Returns: tuple of (str, list) — [0] is the cleared textbox value
    (always ""), [1] is the updated history: a list of (question, answer)
    tuples. The actual answer text is history[-1][-1] after one exchange.

REAL ISSUE FOUND AND FIXED: gradio_client connects ANONYMOUSLY by default,
which gets a much more limited ZeroGPU quota tier on Project 2's Space.
A real test hit "You have exceeded your ZeroGPU quota (180s requested vs.
156s left). Try again in 23:49:03" after only a couple of calls — the
error message itself names the fix: authenticate with an HF token for a
higher quota tier. HF_TOKEN is read from an environment variable (set as
a Streamlit Cloud secret for the real deployment) — see DEPLOY.md.
"""

import logging
import os

from gradio_client import Client

log = logging.getLogger(__name__)

SPACE_ID = "lzhang2026/agri-rag-assistant"
API_NAME = "/ask_question"

_client = None


def _get_client():
    global _client
    if _client is None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            log.warning(
                "HF_TOKEN not set — connecting to the literature search Space "
                "ANONYMOUSLY, which gets a much lower ZeroGPU quota and may hit "
                "'exceeded your ZeroGPU quota' errors quickly. Set HF_TOKEN "
                "(a free token from huggingface.co/settings/tokens) for higher "
                "quota — see DEPLOY.md."
            )
        log.info(f"Connecting to Project 2's live Space: {SPACE_ID} ...")
        _client = Client(SPACE_ID, token=hf_token)
    return _client


def search_literature(question: str) -> str:
    """
    Search agronomic research literature (via Project 2's live deployed
    RAG system) and return a grounded, cited answer.
    """
    try:
        client = _get_client()
        result = client.predict(question=question, history=[], api_name=API_NAME)

        # result = (cleared_textbox_value, updated_history). updated_history
        # is [(question, answer)] after one exchange with history=[] passed
        # in — see module docstring for the confirmed shape.
        _, history = result
        if history:
            return history[-1][-1]
        return "No response received from the literature search service."

    except Exception as e:
        log.error(f"Remote literature search failed: {e}")
        return f"Literature search is currently unavailable: {e}"
