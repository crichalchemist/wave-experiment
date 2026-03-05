"""Tests for dataset sourcing pipeline."""
from unittest.mock import MagicMock


# --- Task 3a: HuggingFace Loader ---

def test_hf_loader_import():
    from src.data.sourcing.hf_loader import load_hf_legal_batch
    assert callable(load_hf_legal_batch)


def test_hf_loader_returns_list_with_mock(monkeypatch):
    """Loader returns list of dicts with required keys (mocked, no network)."""
    from src.data.sourcing import hf_loader

    class FakeDataset:
        def __iter__(self):
            yield {"text": "Court document about financial transfers.", "id": 1}
            yield {"text": "Legal analysis of jurisdiction.", "id": 2}

    monkeypatch.setattr(
        "src.data.sourcing.hf_loader.load_dataset",
        lambda *a, **kw: FakeDataset(),
    )
    batch = hf_loader.load_hf_legal_batch(
        dataset_name="fake/dataset",
        split="train",
        max_documents=5,
        text_field="text",
    )
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert batch[0].text
    assert batch[0].source
    assert isinstance(batch[0].metadata, dict)


def test_hf_loader_metadata_preserved(monkeypatch):
    from src.data.sourcing import hf_loader

    class FakeDataset:
        def __iter__(self):
            yield {"text": "Document text.", "extra_field": "extra_val"}

    monkeypatch.setattr(
        "src.data.sourcing.hf_loader.load_dataset",
        lambda *a, **kw: FakeDataset(),
    )
    batch = hf_loader.load_hf_legal_batch(
        dataset_name="test/dataset",
        config_name="test_config",
        split="train",
        max_documents=3,
        text_field="text",
    )
    assert len(batch) == 1
    meta = batch[0].metadata
    assert meta["dataset"] == "test/dataset"
    assert meta["split"] == "train"
    assert meta["config"] == "test_config"


def test_hf_loader_keyword_filter(monkeypatch):
    from src.data.sourcing import hf_loader

    class FakeDataset:
        def __iter__(self):
            yield {"text": "Financial records from 2003."}
            yield {"text": "Unrelated weather report."}
            yield {"text": "Financial audit of institution."}

    monkeypatch.setattr(
        "src.data.sourcing.hf_loader.load_dataset",
        lambda *a, **kw: FakeDataset(),
    )
    batch = hf_loader.load_hf_legal_batch(
        dataset_name="fake/ds",
        split="train",
        max_documents=10,
        text_field="text",
        keyword_filter="financial",
    )
    assert len(batch) == 2
    assert all("financial" in b.text.lower() for b in batch)


def test_hf_loader_respects_max(monkeypatch):
    from src.data.sourcing import hf_loader

    class FakeDataset:
        def __iter__(self):
            for i in range(100):
                yield {"text": f"Document {i}."}

    monkeypatch.setattr(
        "src.data.sourcing.hf_loader.load_dataset",
        lambda *a, **kw: FakeDataset(),
    )
    batch = hf_loader.load_hf_legal_batch(
        dataset_name="fake/ds", split="train", max_documents=5, text_field="text",
    )
    assert len(batch) == 5


# --- Task 3b: DOJ / CourtListener Loader ---

def test_doj_loader_import():
    from src.data.sourcing.doj_loader import load_courtlistener_batch
    assert callable(load_courtlistener_batch)


def test_doj_loader_returns_list(monkeypatch):
    """Loader returns list even when API returns empty results."""
    from src.data.sourcing import doj_loader

    class _MockResponse:
        def raise_for_status(self): pass
        def json(self): return {"results": [], "count": 0}

    monkeypatch.setattr(
        "src.data.sourcing.doj_loader._httpx_get",
        lambda url, **kw: _MockResponse(),
    )
    results = doj_loader.load_courtlistener_batch(case_name="Maxwell", max_documents=5)
    assert isinstance(results, list)
    assert len(results) == 0


def test_doj_loader_normalizes_fields(monkeypatch):
    from src.data.sourcing import doj_loader

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
    results = doj_loader.load_courtlistener_batch(case_name="Maxwell", max_documents=5)
    assert len(results) == 1
    assert results[0].text == "Court document text here."
    assert results[0].source.startswith("courtlistener:")
    assert results[0].metadata["jurisdiction"] == "SDNY"
    assert results[0].metadata["date_filed"] == "2021-11-01"


# --- Task 3c: International Loader ---

def test_international_loader_import():
    from src.data.sourcing.international_loader import load_occrp_batch
    assert callable(load_occrp_batch)


def test_international_loader_returns_list(monkeypatch):
    from src.data.sourcing import international_loader

    monkeypatch.setattr(
        "src.data.sourcing.international_loader._httpx_get",
        lambda url, **kw: type("R", (), {
            "raise_for_status": lambda s: None,
            "text": "<html><article>Investigation text here</article></html>",
        })(),
    )
    results = international_loader.load_occrp_batch(max_documents=5)
    assert isinstance(results, list)
    assert results == []


def test_github_foia_loader_import():
    from src.data.sourcing.international_loader import load_github_public_foia
    assert callable(load_github_public_foia)
