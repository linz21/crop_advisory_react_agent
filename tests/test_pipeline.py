"""
Tests for the crop advisory agent.
Run:  pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAuditLog:
    def test_log_interaction_writes_valid_json_line(self, tmp_path, monkeypatch):
        from src.guardrails import audit_log
        monkeypatch.setattr(audit_log, "AUDIT_LOG_PATH", tmp_path / "audit_log.jsonl")

        audit_log.log_interaction(
            session_id="test-session", question="Test question?", answer="Test answer.",
            iterations=2, guardrails_passed=True, guardrails_issues=[],
        )

        entries = audit_log.read_audit_log()
        assert len(entries) == 1
        assert entries[0]["session_id"] == "test-session"
        assert entries[0]["question"] == "Test question?"
        assert entries[0]["guardrails_passed"] is True

    def test_read_audit_log_filters_flagged_only(self, tmp_path, monkeypatch):
        from src.guardrails import audit_log
        monkeypatch.setattr(audit_log, "AUDIT_LOG_PATH", tmp_path / "audit_log.jsonl")

        audit_log.log_interaction(
            session_id="s1", question="Q1", answer="A1",
            iterations=1, guardrails_passed=True, guardrails_issues=[],
        )
        audit_log.log_interaction(
            session_id="s2", question="Q2", answer="A2",
            iterations=1, guardrails_passed=False, guardrails_issues=["flagged reason"],
        )

        all_entries = audit_log.read_audit_log(only_flagged=False)
        flagged_only = audit_log.read_audit_log(only_flagged=True)

        assert len(all_entries) == 2
        assert len(flagged_only) == 1
        assert flagged_only[0]["session_id"] == "s2"

    def test_read_audit_log_returns_empty_list_when_no_file(self, tmp_path, monkeypatch):
        from src.guardrails import audit_log
        monkeypatch.setattr(audit_log, "AUDIT_LOG_PATH", tmp_path / "nonexistent.jsonl")
        assert audit_log.read_audit_log() == []


class TestSourceExtraction:
    """
    Regression tests for _extract_sources() — specifically guards against
    the comma-splitting bug found via a real test run, where a paper
    title containing internal commas ("Enhance Soil Quality, Microbial
    Diversity, and Crop Productivity...") got fragmented into multiple
    bogus separate "sources" instead of staying as one title.
    """

    def test_simple_sources_split_correctly(self):
        from src.agent.react_agent import _extract_sources
        obs = "Some answer text.\nSources: Paper One. (2020), Paper Two. (2021)"
        result = _extract_sources(obs)
        assert result == ["Paper One. (2020)", "Paper Two. (2021)"]

    def test_title_with_internal_commas_stays_intact(self):
        from src.agent.react_agent import _extract_sources
        obs = (
            "Sources: Enhancing Soil Quality, Microbial Diversity, and Crop "
            "Productivity in Newly Cultivated Land. (2025), Another Paper. (2026)"
        )
        result = _extract_sources(obs)
        assert len(result) == 2
        assert result[0] == "Enhancing Soil Quality, Microbial Diversity, and Crop Productivity in Newly Cultivated Land. (2025)"
        assert result[1] == "Another Paper. (2026)"

    def test_no_sources_line_returns_empty_list(self):
        from src.agent.react_agent import _extract_sources
        result = _extract_sources("Just some answer text with no sources line.")
        assert result == []


class TestGuardrails:
    def test_valid_response_passes(self):
        from src.guardrails.validators import validate_response
        is_valid, issues = validate_response(
            "Based on the research, nitrogen timing can improve yield in some conditions, "
            "though results vary by region and growing season."
        )
        assert is_valid
        assert issues == []

    def test_empty_response_fails(self):
        from src.guardrails.validators import validate_response
        is_valid, issues = validate_response("")
        assert not is_valid
        assert len(issues) > 0

    def test_risky_absolute_claim_flagged(self):
        from src.guardrails.validators import validate_response
        is_valid, issues = validate_response(
            "This fertilizer rate is guaranteed to increase your yield."
        )
        assert not is_valid
        assert any("guaranteed" in issue.lower() for issue in issues)

    def test_multiple_risky_patterns_all_flagged(self):
        """
        NOTE: the real guardrails-ai framework returns one FailResult per
        VALIDATOR, not one per matched pattern within it (unlike the
        earlier regex-only placeholder). AbsoluteLanguageValidator lists
        ALL matched patterns within its single failure_reason message,
        rather than producing multiple separate issue entries.
        """
        from src.guardrails.validators import validate_response
        is_valid, issues = validate_response(
            "This is always safe to use and definitely will improve your yield."
        )
        assert not is_valid
        assert len(issues) == 1
        assert "always safe to" in issues[0] and "definitely will" in issues[0]

    def test_self_citation_flagged(self):
        """
        Regression test tied to an ACTUAL observed failure: even after
        explicitly instructing the model not to self-cite (real citations
        are code-guaranteed separately), a real test with Qwen3-4B still
        produced an answer ending with a self-generated
        "[Source: search_literature({...})]" marker.
        """
        from src.guardrails.validators import validate_response
        is_valid, issues = validate_response(
            "Nitrogen timing affects corn yield. "
            '[Source: search_literature({"question": "..."})]'
        )
        assert not is_valid
        assert len(issues) > 0

    def test_clean_response_with_code_appended_sources_passes(self):
        """
        Regression test for a real bug caught while writing this test:
        the self-citation validator's "^Sources?:" pattern would incorrectly
        flag react_agent.py's OWN code-appended "Sources consulted:" block
        if validation ran on the combined final text. Fixed by validating
        the model's RAW answer BEFORE sources are appended (see
        react_agent.py's _finalize()) — this test validates the raw
        answer only, matching that corrected call site, and confirms a
        clean answer with no self-citation attempt passes.
        """
        from src.guardrails.validators import validate_response
        is_valid, issues = validate_response(
            "Nitrogen timing affects corn yield through several mechanisms, "
            "including split application timing relative to growth stages."
        )
        assert is_valid
        assert issues == []


class TestObservability:
    def test_verify_langsmith_returns_false_when_unconfigured(self, monkeypatch):
        from src.observability.langsmith_setup import verify_langsmith_configured
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        assert verify_langsmith_configured() is False

    def test_verify_langsmith_returns_true_when_configured(self, monkeypatch):
        from src.observability.langsmith_setup import verify_langsmith_configured
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "fake-key-for-test")
        assert verify_langsmith_configured() is True


class TestMemoryRequiresConfig:
    def test_missing_redis_host_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("REDIS_HOST", raising=False)
        from src.memory.redis_memory import SessionMemory
        with pytest.raises(RuntimeError, match="REDIS_HOST"):
            SessionMemory(session_id="test123")


class TestAgentIntegration:
    """
    Full agent tests (LLM loading + tool calls) are NOT included here —
    loading the local LLM is slow and both tools depend on external state
    (Project 1's live API, Project 3's local retriever/index) that isn't
    guaranteed to be present in a CI environment. This is a documented gap
    (see README) rather than a hidden one — these need manual/integration
    testing, not routine unit testing.
    """
    def test_tools_are_importable(self):
        from src.tools.yield_prediction_tool import predict_corn_yield
        from src.tools.literature_search_tool import search_literature
        assert predict_corn_yield is not None
        assert search_literature is not None
