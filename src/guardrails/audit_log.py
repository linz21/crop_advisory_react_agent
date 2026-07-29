"""
Structured audit logging — records every agent interaction as a single
JSON line, capturing what was asked, what was answered, whether Guardrails
flagged anything, and basic tool-usage metadata. Distinct from Python's
`logging` module's free-text logs (used elsewhere in this project for
debugging) — this is specifically structured, one-record-per-interaction,
and meant to be queryable/auditable after the fact (e.g. "show me every
interaction where Guardrails flagged something this week").

Usage:
    from src.guardrails.audit_log import log_interaction
    log_interaction(
        session_id="...", question="...", answer="...",
        iterations=2, guardrails_passed=True, guardrails_issues=[],
    )
"""

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

AUDIT_LOG_PATH = Path("logs/audit_log.jsonl")


def log_interaction(
    session_id: str,
    question: str,
    answer: str,
    iterations: int,
    guardrails_passed: bool,
    guardrails_issues: list[str],
    llm_provider: str = "local",
):
    """
    Appends one JSON line per interaction. Best-effort — a logging failure
    should never break the actual answer being returned to the user, so
    this catches and logs (via the normal logger) rather than raising.
    """
    record = {
        "timestamp": time.time(),
        "session_id": session_id,
        "question": question,
        "answer_length_chars": len(answer),
        "iterations": iterations,
        "llm_provider": llm_provider,
        "guardrails_passed": guardrails_passed,
        "guardrails_issues": guardrails_issues,
    }

    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.warning(f"Failed to write audit log entry: {e}")


def read_audit_log(only_flagged: bool = False) -> list[dict]:
    """
    Reads back all audit log entries — useful for a quick review of past
    interactions, e.g. to find every case Guardrails flagged something.
    """
    if not AUDIT_LOG_PATH.exists():
        return []

    entries = []
    with open(AUDIT_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if only_flagged and entry.get("guardrails_passed", True):
                continue
            entries.append(entry)
    return entries
