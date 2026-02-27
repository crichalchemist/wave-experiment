"""Test dual output pipeline: evidence + training from scraped documents."""
import pytest
from unittest.mock import patch


def test_process_for_evidence_returns_dict():
    from src.data.sourcing.dual_pipeline import process_for_evidence
    from src.data.sourcing.foia_scraper import FOIADocument

    doc = FOIADocument(
        source_portal="fbi_vault", title="Test",
        url="https://test", date=None, collection=None,
        text="Resource allocation gap in oversight records from 2013-2017.",
        pdf_path=None,
    )

    mock_scores = {"c": 0.7, "kappa": 0.3, "j": 0.3, "p": 0.3,
                   "eps": 0.3, "lam_L": 0.3, "lam_P": 0.5, "xi": 0.6}
    with patch("src.data.sourcing.dual_pipeline.get_construct_scores", return_value=mock_scores):
        result = process_for_evidence(doc)

    assert "construct_scores" in result
    assert "threatened_constructs" in result
    assert "phi" in result
    assert isinstance(result["construct_scores"], dict)
    assert isinstance(result["phi"], float)


def test_process_for_training_returns_dict():
    from src.data.sourcing.dual_pipeline import process_for_training
    from src.data.sourcing.foia_scraper import FOIADocument

    doc = FOIADocument(
        source_portal="fbi_vault", title="Test",
        url="https://test", date=None, collection=None,
        text="Investigation revealed systematic suppression of evidence.",
        pdf_path=None,
    )

    mock_scores = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                   "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.8}
    with patch("src.data.sourcing.dual_pipeline.get_construct_scores", return_value=mock_scores):
        result = process_for_training(doc)

    assert "text" in result
    assert "scores" in result
    assert "metadata" in result
    assert len(result["scores"]) == 8


def test_run_dual_pipeline():
    from src.data.sourcing.dual_pipeline import run_dual_pipeline
    from src.data.sourcing.foia_scraper import FOIADocument

    docs = [
        FOIADocument(
            source_portal="fbi_vault", title="Doc 1",
            url="https://test/1", date=None, collection=None,
            text="Financial records show unexplained gaps in reporting.",
            pdf_path=None,
        ),
    ]

    mock_scores = {"c": 0.5, "kappa": 0.5, "j": 0.5, "p": 0.5,
                   "eps": 0.5, "lam_L": 0.5, "lam_P": 0.5, "xi": 0.5}
    with patch("src.data.sourcing.dual_pipeline.get_construct_scores", return_value=mock_scores):
        result = run_dual_pipeline(docs)

    assert "evidence" in result
    assert "training" in result
    assert len(result["evidence"]) == 1
    assert len(result["training"]) == 1


def test_dual_pipeline_skips_empty_docs():
    from src.data.sourcing.dual_pipeline import run_dual_pipeline
    from src.data.sourcing.foia_scraper import FOIADocument

    docs = [
        FOIADocument(source_portal="fbi_vault", title="Empty", url="https://test",
                     date=None, collection=None, text="", pdf_path=None),
        FOIADocument(source_portal="fbi_vault", title="Whitespace", url="https://test",
                     date=None, collection=None, text="   ", pdf_path=None),
    ]

    result = run_dual_pipeline(docs)
    assert len(result["evidence"]) == 0
    assert len(result["training"]) == 0
