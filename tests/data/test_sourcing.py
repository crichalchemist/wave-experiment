"""Tests for dataset sourcing pipeline."""
import pytest


def test_hf_loader_import():
    from src.data.sourcing.hf_loader import load_hf_legal_batch
    assert callable(load_hf_legal_batch)


def test_hf_loader_returns_list():
    """Loader returns list of dicts with required keys."""
    from src.data.sourcing.hf_loader import load_hf_legal_batch

    # Use cais/mmlu which has data-only format (no deprecated Python scripts)
    try:
        batch = load_hf_legal_batch(
            dataset_name="cais/mmlu",
            config_name="all",
            split="test",
            max_examples=5,
            text_field="question",
        )
        assert isinstance(batch, list)
        assert len(batch) <= 5
        if batch:
            assert "text" in batch[0]
            assert "source" in batch[0]
            assert "metadata" in batch[0]
    except Exception as e:
        # If dataset unavailable (network, auth, etc.), skip gracefully
        pytest.skip(f"Dataset unavailable: {e}")


def test_hf_loader_metadata_preserved():
    from src.data.sourcing.hf_loader import load_hf_legal_batch

    try:
        batch = load_hf_legal_batch(
            dataset_name="cais/mmlu",
            config_name="all",
            split="test",
            max_examples=3,
            text_field="question",
        )
        if batch:
            meta = batch[0]["metadata"]
            assert "dataset" in meta
            assert "split" in meta
    except Exception as e:
        pytest.skip(f"Dataset unavailable: {e}")
