"""
Tests for the crop advisory agent.
Run:  pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


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
        from src.guardrails.validators import validate_response
        is_valid, issues = validate_response(
            "This is always safe to use and definitely will improve your yield."
        )
        assert not is_valid
        assert len(issues) >= 2


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
