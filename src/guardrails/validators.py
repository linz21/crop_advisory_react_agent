"""
Output validation using the REAL guardrails-ai framework (v0.9.2), not a
regex placeholder. Two custom validators, both tied to failures actually
OBSERVED during testing today, not generic examples:

1. AbsoluteLanguageValidator — flags overly confident/unconditional claims
   in agronomic advice (a real farmer could act on this output).

2. SelfCitationValidator — flags the model adding its OWN citation-style
   markers (e.g. "[Source: ...]", "[1]") in the answer text. Even after
   explicitly instructing the model not to self-cite (since sources are
   code-guaranteed separately — see react_agent.py's _finalize()), a real
   test still showed Qwen3-4B occasionally adding
   "[Source: search_literature({...})]" to its own answer text. This
   validator catches that recurrence.

Guardrails Hub validators (pre-built, e.g. ToxicLanguage) require a
separate `guardrails hub install` step and, for some validators, a Hub
account — not used here to avoid an additional account-signup dependency,
consistent with keeping this project's external dependencies minimal.
Custom validators (this file) don't have that requirement.

NOTE on API: guardrails-ai's custom validator method is documented in
places as `_validate` and in other places as `validate` depending on
version/source. Implemented with `validate` below per the majority of
current examples — if you hit a "not implemented" error on your installed
version, try renaming to `_validate` (see guardrails-ai's own source for
your installed version to confirm).

Usage:
    from src.guardrails.validators import build_guard
    guard = build_guard()
    result = guard.validate(answer_text)
    if not result.validation_passed:
        # result.error contains the specific validator's message
"""

import logging
import re
from typing import Any, Dict

from guardrails import Guard, OnFailAction
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

log = logging.getLogger(__name__)


# Phrases that would indicate the agent is giving unsafe/reckless agronomic
# advice without appropriate caveats — a real farmer could act on this
# output, so overly confident absolute claims are worth flagging.
RISKY_PATTERNS = [
    r"\bguaranteed\b",
    r"\bdefinitely will\b",
    r"\balways safe to\b",
    r"\bno need to (check|verify|consult)\b",
]


@register_validator(name="absolute-language", data_type="string")
class AbsoluteLanguageValidator(Validator):
    """
    Flags overly confident, unconditional claims in agronomic advice.
    Migrated from an earlier regex-only placeholder into a real
    guardrails-ai Validator — same underlying patterns, now integrated
    into a proper Guard rather than a standalone function nothing called.
    """

    def __init__(self, on_fail=None):
        super().__init__(on_fail=on_fail)

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        found = [p for p in RISKY_PATTERNS if re.search(p, value, re.IGNORECASE)]
        if found:
            return FailResult(
                error_message=(
                    f"Response contains overly absolute claim(s) matching: {found}. "
                    "Agronomic advice should acknowledge uncertainty rather than "
                    "making unconditional guarantees."
                ),
            )
        return PassResult()


@register_validator(name="self-citation", data_type="string")
class SelfCitationValidator(Validator):
    """
    Flags the model adding its OWN citation-style markers in the answer
    text — e.g. "[Source: ...]", "[1]", a self-generated "Sources:" line.
    The agent's prompt explicitly instructs the model not to do this
    (see react_agent.py's REACT_PROMPT_TEMPLATE), since real citations are
    code-guaranteed separately via _extract_sources()/_finalize(). Tested
    empirically that Claude Sonnet 4.5 respects this instruction reliably;
    a real test with Qwen3-4B still showed an occasional
    "[Source: search_literature({...})]" slipping through despite the
    instruction — this validator catches that recurrence rather than
    trusting the instruction to always hold.
    """

    def __init__(self, on_fail=None):
        super().__init__(on_fail=on_fail)

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        patterns = [r"\[Source:", r"\[\d+\]", r"^Sources?:", r"\[Citation"]
        found = [p for p in patterns if re.search(p, value, re.MULTILINE)]
        if found:
            return FailResult(
                error_message=(
                    f"Response contains self-generated citation markers matching: "
                    f"{found}, despite being instructed not to self-cite. Real "
                    "citations are appended separately by the system."
                ),
            )
        return PassResult()


def build_guard() -> Guard:
    """
    Builds the Guard used to validate agent final answers. on_fail is set
    to OnFailAction.NOOP — validators here are used for DETECTION and
    audit logging (see audit_log.py), not automatic blocking/rewriting,
    since a false positive silently altering a farmer's answer is its own
    risk. Failures are logged and can be reviewed, not auto-corrected.
    """
    return Guard().use(
        AbsoluteLanguageValidator(on_fail=OnFailAction.NOOP),
        SelfCitationValidator(on_fail=OnFailAction.NOOP),
    )


def validate_response(response_text: str) -> tuple[bool, list[str]]:
    """
    Convenience wrapper matching the old placeholder's return signature
    (is_valid, issues) — used by react_agent.py and existing tests, so
    callers don't need to know about Guard's own result object shape.

    NOTE: Guard's top-level `result.error` is None even on failure — the
    actual per-validator failure detail lives in
    `result.validation_summaries[i].failure_reason`, found by directly
    inspecting the result object's attributes (not obvious from the
    handful of docs examples available, which mostly show the simpler
    `guard.validate()` raising an exception path rather than inspecting
    a returned result object).
    """
    if not response_text or not response_text.strip():
        return False, ["Response is empty."]

    guard = build_guard()
    result = guard.validate(response_text)

    if result.validation_passed:
        return True, []

    issues = [
        f"{s.validator_name}: {s.failure_reason}"
        for s in result.validation_summaries
        if s.validator_status == "fail"
    ]
    return False, issues or ["Validation failed (no detail available)."]
