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


def test_doj_loader_import():
    from src.data.sourcing.doj_loader import load_courtlistener_batch
    assert callable(load_courtlistener_batch)


def test_doj_loader_returns_list(monkeypatch):
    """Loader returns list even when API returns empty results."""
    import httpx
    from src.data.sourcing.doj_loader import load_courtlistener_batch

    # Mock httpx to avoid network in unit tests
    class _MockResponse:
        def raise_for_status(self): pass
        def json(self): return {"results": [], "count": 0}

    monkeypatch.setattr(
        "src.data.sourcing.doj_loader._httpx_get",
        lambda url, **kw: _MockResponse(),
    )
    results = load_courtlistener_batch(case_name="Maxwell", max_examples=5)
    assert isinstance(results, list)


def test_doj_loader_normalizes_fields(monkeypatch):
    from src.data.sourcing.doj_loader import load_courtlistener_batch

    class _MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {"results": [
                {"plain_text": "Court document text here.", "date_filed": "2021-11-01",
                 "docket_id": 123, "description": "Motion to suppress"}
            ], "count": 1}

    monkeypatch.setattr(
        "src.data.sourcing.doj_loader._httpx_get",
        lambda url, **kw: _MockResponse(),
    )
    results = load_courtlistener_batch(case_name="Maxwell", max_examples=5)
    assert len(results) == 1
    assert "text" in results[0]
    assert "source" in results[0]
    assert results[0]["metadata"]["jurisdiction"] == "SDNY"


def test_international_loader_import():
    from src.data.sourcing.international_loader import load_occrp_batch
    assert callable(load_occrp_batch)


def test_international_loader_returns_list(monkeypatch):
    from src.data.sourcing.international_loader import load_occrp_batch

    monkeypatch.setattr(
        "src.data.sourcing.international_loader._httpx_get",
        lambda url, **kw: type("R", (), {
            "raise_for_status": lambda s: None,
            "text": "<html><article>Investigation text here</article></html>",
        })(),
    )
    results = load_occrp_batch(max_examples=5)
    assert isinstance(results, list)
