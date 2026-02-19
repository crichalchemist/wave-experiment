"""Unit tests for welfare scoring module."""
from src.inference.welfare_scoring import infer_threatened_constructs, phi_gradient_wrt, score_hypothesis_welfare
from src.detective.hypothesis import Hypothesis
from dataclasses import replace


class TestInferThreatenedConstructs:
    def test_care_pattern(self):
        text = "Temporal gap in resource allocation from 2013-2017"
        constructs = infer_threatened_constructs(text)
        assert "c" in constructs

    def test_protection_pattern(self):
        text = "Redacted correspondence about safeguarding protocols"
        constructs = infer_threatened_constructs(text)
        assert "lam" in constructs

    def test_truth_pattern(self):
        text = "Evidence of systematic suppression of documents"
        constructs = infer_threatened_constructs(text)
        assert "xi" in constructs

    def test_multiple_constructs(self):
        text = "Resource deprivation and ongoing violence against vulnerable populations"
        constructs = infer_threatened_constructs(text)
        assert "c" in constructs
        assert "lam" in constructs

    def test_no_match(self):
        text = "Meeting scheduled for Tuesday"
        assert infer_threatened_constructs(text) == ()


class TestPhiGradient:
    def test_low_value_high_gradient(self):
        gradient = phi_gradient_wrt("c", {"c": 0.1})
        assert gradient > 1.0  # scarce construct → high priority

    def test_high_value_low_gradient(self):
        gradient = phi_gradient_wrt("c", {"c": 0.9})
        assert gradient < 1.0  # low gradient → low priority

    def test_gradient_ordering(self):
        g1 = phi_gradient_wrt("c", {"c": 0.1})
        g2 = phi_gradient_wrt("c", {"c": 0.5})
        g3 = phi_gradient_wrt("c", {"c": 0.9})
        assert g1 > g2 > g3  # monotonic decrease

    def test_floor_prevents_division_by_zero(self):
        gradient = phi_gradient_wrt("c", {"c": 0.0})
        assert gradient < 100  # clamped to prevent inf


class TestScoreHypothesisWelfare:
    def test_high_welfare_when_construct_scarce(self):
        """High welfare score when hypothesis threatens scarce construct."""
        h = Hypothesis.create("Resource allocation gap in 2013-2017", 0.8)
        h = replace(h, threatened_constructs=("c",))

        score = score_hypothesis_welfare(h, {"c": 0.1})
        assert score > 0.5  # high welfare relevance

    def test_low_welfare_when_construct_abundant(self):
        """Low welfare score when hypothesis threatens abundant construct."""
        h = Hypothesis.create("Resource allocation gap in 2013-2017", 0.8)
        h = replace(h, threatened_constructs=("c",))

        score = score_hypothesis_welfare(h, {"c": 0.9})
        assert score < 0.3  # low welfare relevance

    def test_multiple_constructs_sum_gradients(self):
        """Multiple constructs sum their gradients."""
        h = Hypothesis.create("Resource deprivation and violence", 0.8)
        h = replace(h, threatened_constructs=("c", "lam"))

        score_single = score_hypothesis_welfare(
            replace(h, threatened_constructs=("c",)),
            {"c": 0.3, "lam": 0.3}
        )
        score_double = score_hypothesis_welfare(h, {"c": 0.3, "lam": 0.3})

        assert score_double > score_single  # more constructs → higher score

    def test_no_constructs_returns_zero(self):
        """No threatened constructs → welfare score 0."""
        h = Hypothesis.create("Meeting scheduled", 0.9)
        h = replace(h, threatened_constructs=())

        score = score_hypothesis_welfare(h, {})
        assert score == 0.0

    def test_infers_constructs_if_not_set(self):
        """Infers threatened constructs from text if not already set."""
        h = Hypothesis.create("Resource allocation gap", 0.8)
        # threatened_constructs defaults to ()

        score = score_hypothesis_welfare(h, {"c": 0.2})
        assert score > 0.3  # inferred "c" from "resource"
