"""Integration tests for semantic welfare scoring."""
import pytest
from src.detective.hypothesis import Hypothesis
from src.core.types import Gap, GapType
from src.inference.welfare_scoring import (
    score_hypothesis_welfare,
    compute_gap_urgency,
    infer_threatened_constructs,
)


class TestSemanticWelfareIntegration:

    def test_infer_threatened_constructs_returns_valid_constructs(self):
        """All returned constructs are valid 8-construct symbols."""
        constructs = infer_threatened_constructs("Resource allocation gap")
        valid = {"c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"}
        for c in constructs:
            assert c in valid, f"Invalid construct: {c}"

    def test_keyword_fallback_still_works(self):
        """Keyword matching still catches obvious patterns."""
        constructs = infer_threatened_constructs("Resource allocation gap in healthcare")
        assert "c" in constructs  # Care

    def test_protection_patterns_use_lam_P(self):
        """Protection keywords now map to lam_P (not old 'lam')."""
        constructs = infer_threatened_constructs("Violence and harm to communities")
        assert "lam_P" in constructs
        assert "lam" not in constructs  # Old key should NOT appear

    def test_love_patterns_map_to_lam_L(self):
        """New love patterns map to lam_L."""
        constructs = infer_threatened_constructs("Mutual aid networks build community capacity")
        assert "lam_L" in constructs

    def test_phi_gradient_uses_8_constructs(self):
        """phi_gradient_wrt uses equity weights across 8 constructs.

        With c=1.0 and others defaulting to 0.5, c is the *least* deprived
        construct so it gets a small equity weight.  Gradient should still be
        positive and below the old equal-weight value of 1/8.
        """
        from src.inference.welfare_scoring import phi_gradient_wrt
        gradient = phi_gradient_wrt("c", {"c": 1.0})
        assert gradient > 0.0  # always positive
        assert gradient < 1.0 / 8.0  # less than old equal-weight (c is abundant)

    def test_get_construct_scores_returns_8_keys(self):
        """get_construct_scores returns all 8 construct keys."""
        from src.inference.welfare_scoring import get_construct_scores
        scores = get_construct_scores("Resource allocation gap")
        expected_keys = {"c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"}
        assert set(scores.keys()) == expected_keys

    def test_hypothesis_welfare_scoring_with_new_constructs(self):
        """Hypothesis welfare scoring works with lam_P construct."""
        from dataclasses import replace
        h = Hypothesis.create("Violence against communities", 0.8)
        h = replace(h, threatened_constructs=("lam_P",))
        score = score_hypothesis_welfare(h, {"lam_P": 0.1})
        assert score > 0.5  # scarce protection -> high welfare

    def test_gap_urgency_with_new_constructs(self):
        """Gap urgency works with new lam_L and lam_P constructs."""
        gap = Gap(
            type=GapType.TEMPORAL,
            description="Violence and harm",
            confidence=0.9,
            location="doc.pdf",
            threatened_constructs=("lam_P",),
        )
        urgency = compute_gap_urgency(gap, {"lam_P": 0.1})
        assert urgency > 1.0  # high urgency
