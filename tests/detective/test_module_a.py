"""Tests for Module A — cognitive bias detection (provider-based scorer)."""
from __future__ import annotations

import pytest

from src.core.types import AssumptionType
from src.core.providers import MockProvider
from src.detective.module_a import (
    BiasDetection,
    _CONFIDENCE_THRESHOLD,
    _parse_score,
    detect_cognitive_biases,
)


class TestBiasDetection:
    def test_valid_detection(self) -> None:
        d = BiasDetection(
            assumption_type=AssumptionType.COGNITIVE_BIAS,
            score=0.85,
            source_text="some text",
            bias_type="confirmation",
        )
        assert d.score == 0.85
        assert d.bias_type == "confirmation"

    def test_invalid_score_raises(self) -> None:
        with pytest.raises(ValueError, match="score"):
            BiasDetection(
                assumption_type=AssumptionType.COGNITIVE_BIAS,
                score=1.5,
                source_text="text",
                bias_type="confirmation",
            )


class TestParseScore:
    def test_parses_score_colon(self) -> None:
        assert _parse_score("score: 0.85") == 0.85

    def test_parses_confidence_equals(self) -> None:
        assert _parse_score("confidence=0.72") == 0.72

    def test_returns_default_on_no_match(self) -> None:
        assert _parse_score("no number here") == 0.0

    def test_clamps_above_one(self) -> None:
        assert _parse_score("score: 1.5") == 1.0

    def test_clamps_below_zero(self) -> None:
        # Regex only matches positive numbers, so this returns default
        assert _parse_score("score: -0.5") == 0.0


class TestDetectCognitiveBiases:
    def test_confirmation_bias_keyword_only(self) -> None:
        """No provider: keyword match returns score=1.0."""
        results = detect_cognitive_biases(
            "This confirms our hypothesis about the pattern."
        )
        assert len(results) >= 1
        assert all(r.assumption_type == AssumptionType.COGNITIVE_BIAS for r in results)
        assert all(r.score == 1.0 for r in results)

    def test_survivorship_bias_keyword_only(self) -> None:
        results = detect_cognitive_biases(
            "Only those who succeeded are represented in the data."
        )
        assert len(results) >= 1
        assert any(r.bias_type == "survivorship" for r in results)

    def test_anchoring_bias_keyword_only(self) -> None:
        results = detect_cognitive_biases(
            "The initial report set the baseline assumption."
        )
        assert len(results) >= 1
        assert any(r.bias_type == "anchoring" for r in results)

    def test_ingroup_bias_keyword_only(self) -> None:
        results = detect_cognitive_biases(
            "Our group always outperforms them."
        )
        assert len(results) >= 1
        assert any(r.bias_type == "ingroup" for r in results)

    def test_no_bias_detected(self) -> None:
        """Text with no bias patterns returns empty list."""
        results = detect_cognitive_biases("Meeting scheduled for Tuesday.")
        assert results == []

    def test_with_provider_scores_via_llm(self) -> None:
        """When provider given, uses LLM scoring instead of default 1.0."""
        provider = MockProvider(response="cognitive_bias, score: 0.85")
        results = detect_cognitive_biases(
            "This confirms our hypothesis about the trend.",
            provider=provider,
        )
        assert len(results) >= 1
        assert results[0].score == 0.85

    def test_with_provider_below_threshold_filtered(self) -> None:
        """Provider score below threshold is filtered out."""
        provider = MockProvider(response="score: 0.2")
        results = detect_cognitive_biases(
            "This confirms our view.",
            provider=provider,
            threshold=0.5,
        )
        assert results == []

    def test_accepts_provider_kwarg(self) -> None:
        """Module A should accept a ModelProvider like B and C."""
        provider = MockProvider(response="cognitive_bias, score: 0.85")
        results = detect_cognitive_biases(
            "Survivors recall only the successful outcomes, ignoring failures.",
            provider=provider,
        )
        assert isinstance(results, list)
