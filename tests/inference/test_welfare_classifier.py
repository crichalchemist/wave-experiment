"""Tests for semantic welfare classifier."""
import pytest
from unittest.mock import patch, MagicMock
import torch


# These tests work whether or not the model is trained.
# When model doesn't exist, the module returns zero scores (graceful fallback).


class TestWelfareClassifier:
    """Test semantic welfare scoring."""

    def test_get_construct_scores_returns_dict(self):
        """get_construct_scores returns dict with 8 constructs."""
        from src.inference.welfare_classifier import get_construct_scores

        text = "Resource allocation gap in healthcare"
        scores = get_construct_scores(text)

        assert isinstance(scores, dict)
        assert len(scores) == 8

        expected_keys = {"c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"}
        assert set(scores.keys()) == expected_keys

        # All scores should be [0,1]
        for construct, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{construct} score out of range"

    def test_infer_threatened_constructs_threshold(self):
        """Threshold filtering works."""
        from src.inference.welfare_classifier import infer_threatened_constructs

        text = "Matrix of domination structures power relations"

        constructs_low = infer_threatened_constructs(text, threshold=0.2)
        constructs_high = infer_threatened_constructs(text, threshold=0.7)

        assert len(constructs_high) <= len(constructs_low)
        assert all(c in constructs_low for c in constructs_high)

    def test_backward_compatibility(self):
        """infer_threatened_constructs returns tuple of strings."""
        from src.inference.welfare_classifier import infer_threatened_constructs

        text = "Violence and harm to communities"
        constructs = infer_threatened_constructs(text)

        assert isinstance(constructs, tuple)
        assert all(isinstance(c, str) for c in constructs)
        assert all(
            c in {"c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"}
            for c in constructs
        )

    def test_fallback_returns_zero_scores_when_model_missing(self):
        """When the model file doesn't exist, returns all-zero scores."""
        from src.inference.welfare_classifier import get_construct_scores, _load_welfare_classifier

        # Clear any cached classifier
        _load_welfare_classifier.cache_clear()

        scores = get_construct_scores("Any text at all")
        assert all(score == 0.0 for score in scores.values())

    def test_fallback_infer_returns_empty_tuple(self):
        """When the model is missing, infer_threatened_constructs returns ()."""
        from src.inference.welfare_classifier import infer_threatened_constructs, _load_welfare_classifier

        _load_welfare_classifier.cache_clear()

        constructs = infer_threatened_constructs("Violence and harm")
        assert constructs == ()

    def test_construct_names_list(self):
        """CONSTRUCT_NAMES contains exactly the 8 expected constructs."""
        from src.inference.welfare_classifier import CONSTRUCT_NAMES

        expected = ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]
        assert CONSTRUCT_NAMES == expected
