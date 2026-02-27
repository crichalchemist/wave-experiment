"""Test detective-generated scenario extraction pipeline."""
import pytest
from unittest.mock import patch, MagicMock


def test_extract_construct_profiles_returns_list():
    from src.inference.scenario_extraction import extract_construct_profiles

    mock_scores = {
        "c": 0.6, "kappa": 0.3, "j": 0.4, "p": 0.5,
        "eps": 0.2, "lam_L": 0.7, "lam_P": 0.3, "xi": 0.5,
    }

    corpus_text = "Sample text about resource allocation. " * 100

    with patch("src.inference.scenario_extraction.get_construct_scores", return_value=mock_scores):
        profiles = extract_construct_profiles(corpus_text)

    assert isinstance(profiles, list)
    assert len(profiles) >= 1
    assert all("scores" in p for p in profiles)
    assert all("chunk_index" in p for p in profiles)
    assert all(len(p["scores"]) == 8 for p in profiles)


def test_extract_profiles_chunks_corpus():
    from src.inference.scenario_extraction import extract_construct_profiles

    corpus_text = ("word " * 500) * 4  # 2000 words → ~4 chunks

    call_count = 0
    def counting_scorer(text):
        nonlocal call_count
        call_count += 1
        return {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    with patch("src.inference.scenario_extraction.get_construct_scores", side_effect=counting_scorer):
        profiles = extract_construct_profiles(corpus_text)

    assert call_count >= 3


def test_extract_profiles_skips_tiny_trailing_chunks():
    from src.inference.scenario_extraction import extract_construct_profiles

    # 550 words → 1 full chunk + 50-word trailing chunk (should be skipped)
    corpus_text = "word " * 550

    call_count = 0
    def counting_scorer(text):
        nonlocal call_count
        call_count += 1
        return {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}

    with patch("src.inference.scenario_extraction.get_construct_scores", side_effect=counting_scorer):
        profiles = extract_construct_profiles(corpus_text)

    assert call_count == 1  # trailing 50-word chunk skipped


def test_identify_trajectory_patterns():
    from src.inference.scenario_extraction import identify_trajectory_patterns

    profiles = [
        {"chunk_index": i, "scores": {
            "c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
            "eps": 0.5, "lam_L": 0.7 - i * 0.1, "lam_P": 0.5, "xi": 0.5,
        }}
        for i in range(5)
    ]

    patterns = identify_trajectory_patterns(profiles)
    assert isinstance(patterns, list)
    assert len(patterns) >= 1
    assert all("label" in p for p in patterns)
    assert all("start_levels" in p for p in patterns)
    assert all("end_levels" in p for p in patterns)


def test_identify_patterns_needs_min_run_length():
    from src.inference.scenario_extraction import identify_trajectory_patterns

    # Only 2 profiles — below default min_run_length=3
    profiles = [
        {"chunk_index": 0, "scores": {"c": 0.8, "kappa": 0.5, "j": 0.5, "p": 0.5,
                                       "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}},
        {"chunk_index": 1, "scores": {"c": 0.2, "kappa": 0.5, "j": 0.5, "p": 0.5,
                                       "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}},
    ]

    patterns = identify_trajectory_patterns(profiles)
    assert patterns == []


def test_identify_patterns_deduplicates():
    from src.inference.scenario_extraction import identify_trajectory_patterns

    # Same decline across overlapping windows should produce only one pattern
    profiles = [
        {"chunk_index": i, "scores": {
            "c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
            "eps": 0.5, "lam_L": 0.8 - i * 0.1, "lam_P": 0.5, "xi": 0.5,
        }}
        for i in range(6)
    ]

    patterns = identify_trajectory_patterns(profiles)
    lam_L_patterns = [p for p in patterns if p["dominant_construct"] == "lam_L"]
    assert len(lam_L_patterns) == 1  # deduplicated


def test_generate_from_template():
    import pandas as pd
    from src.inference.scenario_extraction import generate_from_template

    template = {
        "label": "declining_love",
        "start_levels": {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                         "eps": 0.5, "lam_L": 0.7, "lam_P": 0.5, "xi": 0.5},
        "end_levels": {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                       "eps": 0.5, "lam_L": 0.2, "lam_P": 0.5, "xi": 0.5},
    }

    df = generate_from_template(template, length=200)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert "phi" in df.columns
    assert all(c in df.columns for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"])
    assert df["lam_L"].iloc[0] > df["lam_L"].iloc[-1]  # declining


def test_generate_from_template_phi_positive():
    from src.inference.scenario_extraction import generate_from_template

    template = {
        "label": "test",
        "start_levels": {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                         "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5},
        "end_levels": {"c": 0.3, "kappa": 0.3, "j": 0.3, "p": 0.3,
                       "eps": 0.3, "lam_L": 0.3, "lam_P": 0.3, "xi": 0.3},
    }

    df = generate_from_template(template, length=100)
    assert all(df["phi"] > 0)


def test_generate_from_template_values_clipped():
    from src.inference.scenario_extraction import generate_from_template

    template = {
        "label": "extreme",
        "start_levels": {"c": 0.99, "kappa": 0.99, "j": 0.99, "p": 0.99,
                         "eps": 0.99, "lam_L": 0.99, "lam_P": 0.99, "xi": 0.99},
        "end_levels": {"c": 0.01, "kappa": 0.01, "j": 0.01, "p": 0.01,
                       "eps": 0.01, "lam_L": 0.01, "lam_P": 0.01, "xi": 0.01},
    }

    df = generate_from_template(template, length=200)
    for c in ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]:
        assert df[c].min() >= 0.0
        assert df[c].max() <= 1.0


def test_run_extraction_pipeline():
    from src.inference.scenario_extraction import run_extraction_pipeline
    from unittest.mock import patch
    import tempfile, os

    corpus = "Resource allocation suffered as community bonds weakened. " * 200
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(corpus)
        corpus_path = f.name

    try:
        mock_scores = {"c": 0.7, "kappa": 0.3, "j": 0.4, "p": 0.5,
                       "eps": 0.2, "lam_L": 0.3, "lam_P": 0.3, "xi": 0.5}
        with patch("src.inference.scenario_extraction.get_construct_scores", return_value=mock_scores):
            result = run_extraction_pipeline(corpus_path)

        assert "profiles" in result
        assert "patterns" in result
        assert "scenarios" in result
        assert isinstance(result["profiles"], list)
        assert isinstance(result["patterns"], list)
        assert isinstance(result["scenarios"], list)
    finally:
        os.unlink(corpus_path)
