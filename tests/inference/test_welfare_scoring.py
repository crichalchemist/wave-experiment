"""Unit tests for welfare scoring module."""
from unittest.mock import patch
from src.inference.welfare_scoring import infer_threatened_constructs, phi_gradient_wrt, score_hypothesis_welfare, compute_gap_urgency
from src.detective.hypothesis import Hypothesis
from src.core.types import Gap, GapType
from dataclasses import replace

# Force keyword fallback for tests that verify keyword-based logic.
# Semantic classifier is tested separately in test_welfare_integration_semantic.py.
_force_keyword = patch(
    'src.inference.welfare_classifier.get_construct_scores',
    side_effect=FileNotFoundError("mocked out"),
)


class TestInferThreatenedConstructs:
    @_force_keyword
    def test_care_pattern(self, _mock):
        text = "Temporal gap in resource allocation from 2013-2017"
        constructs = infer_threatened_constructs(text)
        assert "c" in constructs

    @_force_keyword
    def test_protection_pattern(self, _mock):
        text = "Redacted correspondence about safeguarding protocols"
        constructs = infer_threatened_constructs(text)
        assert "lam_P" in constructs

    @_force_keyword
    def test_truth_pattern(self, _mock):
        text = "Evidence of systematic suppression of documents"
        constructs = infer_threatened_constructs(text)
        assert "xi" in constructs

    @_force_keyword
    def test_multiple_constructs(self, _mock):
        text = "Resource deprivation and ongoing violence against vulnerable populations"
        constructs = infer_threatened_constructs(text)
        assert "c" in constructs
        assert "lam_P" in constructs

    @_force_keyword
    def test_no_match(self, _mock):
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
        h = replace(h, threatened_constructs=("c", "lam_P"))

        score_single = score_hypothesis_welfare(
            replace(h, threatened_constructs=("c",)),
            {"c": 0.3, "lam_P": 0.3}
        )
        score_double = score_hypothesis_welfare(h, {"c": 0.3, "lam_P": 0.3})

        assert score_double > score_single  # more constructs → higher score

    @_force_keyword
    def test_no_constructs_returns_zero(self, _mock):
        """No threatened constructs → welfare score 0 (keyword fallback)."""
        h = Hypothesis.create("Meeting scheduled", 0.9)
        h = replace(h, threatened_constructs=())

        score = score_hypothesis_welfare(h, {})
        assert score == 0.0

    @_force_keyword
    def test_infers_constructs_if_not_set(self, _mock):
        """Infers threatened constructs from text if not already set."""
        h = Hypothesis.create("Resource allocation gap", 0.8)
        # threatened_constructs defaults to ()

        score = score_hypothesis_welfare(h, {"c": 0.2})
        assert score > 0.3  # inferred "c" from "resource"


class TestComputeGapUrgency:
    def test_urgency_high_when_confidence_and_welfare_both_high(self):
        """High urgency when gap has high confidence and threatens scarce construct."""
        gap = Gap(
            type=GapType.TEMPORAL,
            description="Resource allocation gap 2013-2017",
            confidence=0.9,
            location="financial_records.pdf",
            threatened_constructs=("c",),
        )

        urgency = compute_gap_urgency(gap, {"c": 0.1})  # scarce
        assert urgency > 1.0  # high urgency (gradient ~1.43 * confidence 0.9 ≈ 1.29)

    @_force_keyword
    def test_urgency_low_when_welfare_low(self, _mock):
        """Low urgency when gap has no welfare threat (keyword fallback)."""
        gap = Gap(
            type=GapType.EVIDENTIAL,
            description="Typo in document",
            confidence=0.95,
            location="memo.txt",
            threatened_constructs=(),
        )

        urgency = compute_gap_urgency(gap, {})
        assert urgency == 0.0

    def test_urgency_scales_with_confidence(self):
        """Urgency scales with epistemic confidence."""
        gap_high_conf = Gap(
            type=GapType.TEMPORAL,
            description="Resource gap",
            confidence=0.9,
            location="doc.pdf",
            threatened_constructs=("c",),
        )
        gap_low_conf = Gap(
            type=GapType.TEMPORAL,
            description="Resource gap",
            confidence=0.3,
            location="doc.pdf",
            threatened_constructs=("c",),
        )

        phi_metrics = {"c": 0.3}
        urgency_high = compute_gap_urgency(gap_high_conf, phi_metrics)
        urgency_low = compute_gap_urgency(gap_low_conf, phi_metrics)

        assert urgency_high > urgency_low

    @_force_keyword
    def test_urgency_infers_constructs_if_not_set(self, _mock):
        """Infers threatened constructs from description if not set."""
        gap = Gap(
            type=GapType.TEMPORAL,
            description="Evidence of resource deprivation",
            confidence=0.8,
            location="doc.pdf",
            # threatened_constructs defaults to ()
        )

        urgency = compute_gap_urgency(gap, {"c": 0.2})
        assert urgency >= 0.5  # inferred "c" from "resource" (1/8 / 0.2 * 0.8 = 0.5)


class TestRecoveryAwareInput:
    """Recovery-aware floor function: x_tilde_i."""

    def test_above_floor_returns_raw(self):
        """Construct above its floor passes through unchanged."""
        from src.inference.welfare_scoring import recovery_aware_input
        assert recovery_aware_input(x_i=0.5, floor_i=0.20, dx_dt_i=0.0, lam_L=0.5) == 0.5

    def test_healing_high_trajectory_high_community(self):
        """Below floor + positive dx/dt + high lam_L -> near floor."""
        from src.inference.welfare_scoring import recovery_aware_input
        result = recovery_aware_input(x_i=0.05, floor_i=0.20, dx_dt_i=0.5, lam_L=0.8)
        assert result > 0.15  # recovery potential lifts toward floor

    def test_intervention_pending_stagnant_with_community(self):
        """Below floor + dx/dt~0 + high lam_L -> moderate recovery potential."""
        from src.inference.welfare_scoring import recovery_aware_input
        result = recovery_aware_input(x_i=0.05, floor_i=0.20, dx_dt_i=0.0, lam_L=0.8)
        assert result > 0.05  # community compensates even when stagnant
        assert result < 0.20  # but doesn't reach floor without trajectory

    def test_true_collapse_stagnant_no_community(self):
        """Below floor + dx/dt~0 + low lam_L -> near raw value (true collapse)."""
        from src.inference.welfare_scoring import recovery_aware_input
        result = recovery_aware_input(x_i=0.05, floor_i=0.20, dx_dt_i=0.0, lam_L=0.05)
        assert result < 0.08  # barely above raw — white supremacy signature

    def test_floor_values_match_design(self):
        """Hard floors per construct match the design doc."""
        from src.inference.welfare_scoring import CONSTRUCT_FLOORS
        assert CONSTRUCT_FLOORS["c"] == 0.20
        assert CONSTRUCT_FLOORS["kappa"] == 0.20
        assert CONSTRUCT_FLOORS["lam_P"] == 0.20
        assert CONSTRUCT_FLOORS["lam_L"] == 0.15
        assert CONSTRUCT_FLOORS["xi"] == 0.30
        assert CONSTRUCT_FLOORS["j"] == 0.10
        assert CONSTRUCT_FLOORS["p"] == 0.10
        assert CONSTRUCT_FLOORS["eps"] == 0.10
