"""
Guardrails AI validation — checks agent output before returning it to the
user, using Guardrails AI (open source, free, no API key required for
basic validators).

STATUS: basic structural/safety validators implemented. NOT yet validated
against real agent outputs — see README Known Gaps.

Usage:
    from src.guardrails.validators import validate_response
    is_valid, issues = validate_response(agent_answer_text)
"""

import logging
import re

log = logging.getLogger(__name__)


# Phrases that would indicate the agent is giving unsafe/reckless agronomic
# advice without appropriate caveats — a real farmer could act on this
# output, so overly confident absolute claims about e.g. chemical
# application rates are worth flagging for human review rather than
# auto-returning.
RISKY_PATTERNS = [
    r"\bguaranteed\b",
    r"\bdefinitely will\b",
    r"\balways safe to\b",
    r"\bno need to (check|verify|consult)\b",
]


def validate_response(response_text: str) -> tuple[bool, list[str]]:
    """
    Runs basic validation checks on the agent's final response text.

    Returns:
        (is_valid, issues) — is_valid is False if any check fails; issues
        lists the specific problems found (empty list if valid).
    """
    issues = []

    if not response_text or not response_text.strip():
        issues.append("Response is empty.")

    for pattern in RISKY_PATTERNS:
        if re.search(pattern, response_text, re.IGNORECASE):
            issues.append(
                f"Response contains an overly absolute claim matching pattern "
                f"'{pattern}' — agronomic advice should acknowledge uncertainty "
                f"rather than making unconditional guarantees."
            )

    # TODO (see README Known Gaps): integrate Guardrails AI's actual
    # validator framework (guardrails-ai package) for more sophisticated
    # checks — e.g. structured output validation for tool-call formatting,
    # PII detection, or domain-specific agronomic safety rules — rather
    # than this regex-based placeholder. This was scoped as a starting
    # point to have SOME validation layer in place, not the final version.

    is_valid = len(issues) == 0
    return is_valid, issues
