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


class TestEquityWeights:
    """Inverse-deprivation weights: w_i = (1/x_i) / sum(1/x_j)."""

    def test_equal_inputs_equal_weights(self):
        """When all constructs equal, weights are equal (1/8)."""
        from src.inference.welfare_scoring import equity_weights
        metrics = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        weights = equity_weights(metrics)
        for w in weights.values():
            assert abs(w - 1/8) < 0.001

    def test_deprived_construct_gets_higher_weight(self):
        """Most deprived construct gets highest weight."""
        from src.inference.welfare_scoring import equity_weights
        metrics = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        metrics["c"] = 0.1  # care is deprived
        weights = equity_weights(metrics)
        assert weights["c"] > weights["kappa"]
        assert weights["c"] > 0.3  # significant shift toward care

    def test_weights_sum_to_one(self):
        """Weights always sum to 1."""
        from src.inference.welfare_scoring import equity_weights
        metrics = {"c": 0.1, "kappa": 0.3, "j": 0.5, "p": 0.7,
                   "eps": 0.2, "lam_L": 0.4, "lam_P": 0.6, "xi": 0.8}
        weights = equity_weights(metrics)
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_care_at_01_others_05_care_weight_dominates(self):
        """c=0.1, others=0.5 -> care weight ~0.42."""
        from src.inference.welfare_scoring import equity_weights
        metrics = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        metrics["c"] = 0.1
        weights = equity_weights(metrics)
        # 1/0.1 = 10, 7*(1/0.5) = 14, total=24, c weight = 10/24 ~ 0.417
        assert weights["c"] > 0.4


class TestCommunityMultiplier:
    """Community solidarity multiplier: f(lam_L) = lam_L^0.5."""

    def test_full_community(self):
        from src.inference.welfare_scoring import community_multiplier
        assert abs(community_multiplier(1.0) - 1.0) < 0.001

    def test_half_community(self):
        """lam_L=0.5 -> f=0.707 (29% diminished)."""
        from src.inference.welfare_scoring import community_multiplier
        result = community_multiplier(0.5)
        assert abs(result - 0.707) < 0.01

    def test_quarter_community(self):
        """lam_L=0.25 -> f=0.5 (50% degradation)."""
        from src.inference.welfare_scoring import community_multiplier
        result = community_multiplier(0.25)
        assert abs(result - 0.5) < 0.01

    def test_near_collapse(self):
        """lam_L=0.04 -> f=0.2 (80% degradation)."""
        from src.inference.welfare_scoring import community_multiplier
        result = community_multiplier(0.04)
        assert abs(result - 0.2) < 0.01

    def test_verification_criterion_2(self):
        """Design doc criterion 2: Phi at lam_L=0.1 < 50% of Phi at lam_L=0.8."""
        from src.inference.welfare_scoring import community_multiplier
        low = community_multiplier(0.1)
        high = community_multiplier(0.8)
        assert low < 0.5 * high


class TestUbuntuSynergy:
    """Psi_ubuntu: construct pairs gain meaning through relationships."""

    def test_balanced_pairs_above_one(self):
        """Balanced pairs produce synergy > 1."""
        from src.inference.welfare_scoring import ubuntu_synergy
        metrics = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        result = ubuntu_synergy(metrics)
        assert result > 1.0

    def test_zero_construct_kills_its_pair(self):
        """A zeroed construct kills its paired synergy contribution."""
        from src.inference.welfare_scoring import ubuntu_synergy
        balanced = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        zeroed = dict(balanced)
        zeroed["c"] = 0.0  # kills c*lam_L pair
        assert ubuntu_synergy(balanced) > ubuntu_synergy(zeroed)

    def test_verification_criterion_5(self):
        """Design doc criterion 5: balanced pairs score higher than unbalanced at same average."""
        from src.inference.welfare_scoring import ubuntu_synergy
        balanced = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        unbalanced = dict(balanced)
        unbalanced["c"] = 0.9
        unbalanced["lam_L"] = 0.1  # same sum but unbalanced
        assert ubuntu_synergy(balanced) > ubuntu_synergy(unbalanced)


class TestDivergencePenalty:
    """Psi_penalty: penalize care-without-love and similar mismatches."""

    def test_no_penalty_when_balanced(self):
        """Balanced pairs have zero penalty."""
        from src.inference.welfare_scoring import divergence_penalty
        metrics = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        assert divergence_penalty(metrics) == 0.0

    def test_paternalism_penalty(self):
        """High care + low love triggers penalty (paternalism detection)."""
        from src.inference.welfare_scoring import divergence_penalty
        metrics = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        metrics["c"] = 0.9
        metrics["lam_L"] = 0.1
        assert divergence_penalty(metrics) > 0.0

    def test_verification_criterion_6(self):
        """Design doc criterion 6: high c + low lam_L scores lower than moderate c + moderate lam_L."""
        from src.inference.welfare_scoring import divergence_penalty
        paternalistic = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        paternalistic["c"] = 0.9
        paternalistic["lam_L"] = 0.1
        moderate = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        assert divergence_penalty(paternalistic) > divergence_penalty(moderate)

    def test_penalty_bounded(self):
        """Penalty is bounded below 1."""
        from src.inference.welfare_scoring import divergence_penalty
        extreme = {"c": 1.0, "kappa": 1.0, "j": 1.0, "p": 0.0,
                   "eps": 0.0, "lam_L": 0.0, "lam_P": 0.0, "xi": 0.0}
        assert divergence_penalty(extreme) < 1.0


class TestPhiGradientEquity:
    """Equity-weighted gradient tests (post-revision)."""

    def test_verification_criterion_1(self):
        """Design doc criterion 1: c=0.1, others=0.5 -> c's gradient >5x any other."""
        metrics = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        metrics["c"] = 0.1
        g_care = phi_gradient_wrt("c", metrics)
        g_others = [phi_gradient_wrt(c, metrics) for c in ["kappa", "j", "p", "eps", "lam_P", "xi"]]
        for g_other in g_others:
            assert g_care > 5 * g_other, f"care gradient {g_care} not >5x {g_other}"

    def test_community_degrades_gradient(self):
        """Low lam_L reduces all gradients via solidarity multiplier."""
        metrics_high = {c: 0.3 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        metrics_high["lam_L"] = 0.8
        metrics_low = dict(metrics_high)
        metrics_low["lam_L"] = 0.1

        g_high = phi_gradient_wrt("c", metrics_high)
        g_low = phi_gradient_wrt("c", metrics_low)
        assert g_high > g_low

    def test_symmetric_case_similar_ordering(self):
        """Verification criterion 9: symmetric case produces similar ordering."""
        metrics = {c: 0.5 for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]}
        gradients = {c: phi_gradient_wrt(c, metrics) for c in ["c", "kappa", "j", "p", "eps", "lam_P", "xi"]}
        values = list(gradients.values())
        assert max(values) / min(values) < 1.1
