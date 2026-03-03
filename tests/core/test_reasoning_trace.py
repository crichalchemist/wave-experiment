"""Tests for ReasoningTrace dataclass and module classification."""

import uuid

from src.core.reasoning_trace import ReasoningTrace, classify_module, try_parse_score


# ---------------------------------------------------------------------------
# classify_module — one test per call site
# ---------------------------------------------------------------------------


class TestClassifyModule:
    def test_module_a_cognitive_bias(self) -> None:
        prompt = (
            "Rate the cognitive bias in the following text on a scale of 0.0 to 1.0.\n"
            "Bias type: confirmation\nText: example\n\n"
            "Reply with: score: <float between 0 and 1>"
        )
        assert classify_module(prompt) == "module_a"

    def test_module_b_historical_determinism(self) -> None:
        prompt = (
            "You are evaluating whether the following text span exhibits "
            "'historical determinism' — the assumption that documentary sequence "
            "reflects causal sequence.\n\nReply with ONLY: score: <float>\n\n"
            "Text span: test\nFull context: test"
        )
        assert classify_module(prompt) == "module_b"

    def test_module_c_geopolitical_presumption(self) -> None:
        prompt = (
            "You are evaluating whether the following sentence contains a "
            "'geopolitical presumption' — an unstated assumption.\n\n"
            "Reply with ONLY: score: <float>\n\nSentence: test\nIdentified actor: UN"
        )
        assert classify_module(prompt) == "module_c"

    def test_evolution_updated_confidence(self) -> None:
        prompt = (
            "Hypothesis: test\nCurrent confidence: 0.50\nNew evidence: data\n\n"
            "How does this evidence affect the hypothesis? "
            "Reply with ONLY a float between 0.0 and 1.0 representing the updated confidence."
        )
        assert classify_module(prompt) == "evolution"

    def test_graph_plausibility(self) -> None:
        prompt = (
            "Score the plausibility of this relationship on a scale from 0.0 to 1.0. "
            "Return only a single float.\n\nRelationship: A → B\nContext: finance"
        )
        assert classify_module(prompt) == "graph"

    def test_parallel_evolution_hypothesis(self) -> None:
        prompt = (
            "You are evolving a hypothesis based on a specific piece of evidence.\n\n"
            "Current hypothesis: test\nCurrent confidence: 0.50\n\n"
            "New evidence: data\n\nHow does this evidence change the hypothesis?"
        )
        assert classify_module(prompt) == "parallel_evolution"

    def test_unknown_for_unrecognized_prompt(self) -> None:
        prompt = "What is the meaning of life?"
        assert classify_module(prompt) == "unknown"


# ---------------------------------------------------------------------------
# try_parse_score
# ---------------------------------------------------------------------------


class TestTryParseScore:
    def test_score_prefix(self) -> None:
        response = "Okay so I think... score: 0.75"
        assert try_parse_score(response) == 0.75

    def test_score_prefix_with_cot(self) -> None:
        response = (
            "Let me evaluate this text carefully. The bias type is confirmation bias, "
            "and the text shows moderate evidence of cherry-picking data.\n\n"
            "score: 0.6"
        )
        assert try_parse_score(response) == 0.6

    def test_confidence_prefix(self) -> None:
        response = "confirmed\nconfidence: 0.82\nThe evidence strongly supports."
        assert try_parse_score(response) == 0.82

    def test_bare_float_at_end(self) -> None:
        response = "After analysis...\n0.45"
        assert try_parse_score(response) == 0.45

    def test_no_score_returns_none(self) -> None:
        response = "This is a detailed analysis with no score."
        assert try_parse_score(response) is None

    def test_integer_score(self) -> None:
        response = "score: 1"
        assert try_parse_score(response) == 1.0

    def test_zero_score(self) -> None:
        response = "score: 0.0"
        assert try_parse_score(response) == 0.0


# ---------------------------------------------------------------------------
# ReasoningTrace.create factory
# ---------------------------------------------------------------------------


class TestReasoningTraceCreate:
    def test_creates_with_auto_fields(self) -> None:
        trace = ReasoningTrace.create(
            prompt="Rate the cognitive bias in this text...",
            raw_response="score: 0.5",
            model="deepseek-r1-distill",
            route="scoring",
            duration_ms=150,
        )
        # Auto-generated fields
        uuid.UUID(trace.id)  # validates it's a real UUID
        assert "T" in trace.timestamp  # ISO-8601
        assert trace.module == "module_a"  # prompt contains "cognitive bias"
        assert trace.parsed_score == 0.5
        # Passed-through fields
        assert trace.model == "deepseek-r1-distill"
        assert trace.route == "scoring"
        assert trace.duration_ms == 150

    def test_frozen_immutability(self) -> None:
        trace = ReasoningTrace.create(
            prompt="test",
            raw_response="score: 0.5",
            model="test",
            route="scoring",
            duration_ms=100,
        )
        import pytest
        with pytest.raises(AttributeError):
            trace.module = "hacked"  # type: ignore[misc]
