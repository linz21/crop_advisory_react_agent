"""
LangSmith tracing setup — enables observability into the agent's reasoning
steps (which tools it called, in what order, with what inputs/outputs)
using LangSmith's free Developer tier (5,000 traces/month, 1 seat, no
credit card required).

LangSmith tracing activates automatically via environment variables once
set — this module just validates they're present and gives a clear error
if not, rather than silently running without tracing.

Usage:
    from src.observability.langsmith_setup import verify_langsmith_configured
    verify_langsmith_configured()  # call once at startup
"""

import logging
import os

log = logging.getLogger(__name__)


def verify_langsmith_configured() -> bool:
    """
    Checks whether LangSmith tracing is properly configured via environment
    variables. Does not raise — tracing is a nice-to-have for observability,
    not a hard requirement for the agent to function, so a missing config
    just logs a warning and returns False rather than blocking anything.
    """
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    api_key = os.getenv("LANGCHAIN_API_KEY")

    if not tracing_enabled or not api_key:
        log.warning(
            "LangSmith tracing is not configured (LANGCHAIN_TRACING_V2 and/or "
            "LANGCHAIN_API_KEY not set). The agent will still work, but you "
            "won't get visibility into its reasoning steps. See README Setup "
            "section — set these two environment variables plus "
            "LANGCHAIN_PROJECT to enable tracing (free tier, no credit card)."
        )
        return False

    log.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    return True
