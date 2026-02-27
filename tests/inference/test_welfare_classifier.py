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
        """When both Hub and local model fail, returns all-zero scores."""
        from pathlib import Path
        from src.inference.welfare_classifier import get_construct_scores, _load_welfare_classifier

        _load_welfare_classifier.cache_clear()

        def _raise_os_error(*args, **kwargs):
            raise OSError("Model not available")

        with patch('src.inference.welfare_classifier.pipeline', _raise_os_error):
            with patch('src.inference.welfare_classifier.MODEL_PATH', Path('/tmp/nonexistent_model')):
                scores = get_construct_scores("Any text at all")
                assert all(score == 0.0 for score in scores.values())

        _load_welfare_classifier.cache_clear()

    def test_fallback_infer_returns_empty_tuple(self):
        """When both Hub and local model are missing, infer_threatened_constructs returns ()."""
        from pathlib import Path
        from src.inference.welfare_classifier import infer_threatened_constructs, _load_welfare_classifier

        _load_welfare_classifier.cache_clear()

        def _raise_os_error(*args, **kwargs):
            raise OSError("Model not available")

        with patch('src.inference.welfare_classifier.pipeline', _raise_os_error):
            with patch('src.inference.welfare_classifier.MODEL_PATH', Path('/tmp/nonexistent_model')):
                constructs = infer_threatened_constructs("Violence and harm")
                assert constructs == ()

        _load_welfare_classifier.cache_clear()

    def test_construct_names_list(self):
        """CONSTRUCT_NAMES contains exactly the 8 expected constructs."""
        from src.inference.welfare_classifier import CONSTRUCT_NAMES

        expected = ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]
        assert CONSTRUCT_NAMES == expected


class TestHubLoading:
    """Test Hub-first loading with local fallback."""

    def test_hub_model_id_is_configured(self):
        """HUB_MODEL_ID constant points to the correct Hub repo."""
        from src.inference.welfare_classifier import HUB_MODEL_ID
        assert HUB_MODEL_ID == "crichalchemist/welfare-constructs-distilbert"

    def test_load_tries_hub_first(self, monkeypatch):
        """_load_welfare_classifier attempts Hub loading before local."""
        import src.inference.welfare_classifier as wc
        wc._load_welfare_classifier.cache_clear()

        mock_pipeline = MagicMock(return_value=[
            [{"label": f"LABEL_{i}", "score": 0.5} for i in range(8)]
        ])

        with patch("src.inference.welfare_classifier.pipeline", mock_pipeline):
            wc._load_welfare_classifier()
            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args
            assert "crichalchemist/welfare-constructs-distilbert" in str(call_args)

        wc._load_welfare_classifier.cache_clear()

    def test_fallback_to_local_when_hub_unavailable(self, monkeypatch):
        """When Hub loading fails, falls back to local model path."""
        import src.inference.welfare_classifier as wc
        wc._load_welfare_classifier.cache_clear()

        call_count = 0
        def mock_pipeline_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Hub unreachable")
            return MagicMock()

        with patch("src.inference.welfare_classifier.pipeline", mock_pipeline_fn):
            with patch("src.inference.welfare_classifier.MODEL_PATH") as mock_path:
                mock_config = MagicMock()
                mock_config.exists.return_value = True
                mock_path.__truediv__ = MagicMock(return_value=mock_config)
                mock_path.__str__ = MagicMock(return_value="models/welfare-constructs-distilbert")
                try:
                    wc._load_welfare_classifier()
                except Exception:
                    pass
                assert call_count >= 2

        wc._load_welfare_classifier.cache_clear()

    def test_get_construct_scores_returns_zeros_when_no_model(self):
        """get_construct_scores returns all zeros when no model is available anywhere."""
        from pathlib import Path
        import src.inference.welfare_classifier as wc
        wc._load_welfare_classifier.cache_clear()

        def _raise_os_error(*args, **kwargs):
            raise OSError("Model not available")

        with patch("src.inference.welfare_classifier.pipeline", _raise_os_error):
            with patch("src.inference.welfare_classifier.MODEL_PATH", Path("/tmp/nonexistent_model")):
                scores = wc.get_construct_scores("test text")
                assert len(scores) == 8
                assert all(v == 0.0 for v in scores.values())

        wc._load_welfare_classifier.cache_clear()
