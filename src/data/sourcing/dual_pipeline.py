"""
Dual output pipeline: process scraped documents for both investigation and training.

Evidence output: welfare construct scores, threatened constructs, Phi value
Training output: welfare-scored examples for classifier + DPO pair generation
"""
from __future__ import annotations

from typing import Any
import logging

from src.data.sourcing.foia_scraper import FOIADocument
from src.inference.welfare_scoring import (
    get_construct_scores, infer_threatened_constructs,
    compute_phi, ALL_CONSTRUCTS,
)

_logger = logging.getLogger(__name__)


def process_for_evidence(doc: FOIADocument) -> dict[str, Any]:
    """Process a document for the investigation evidence pipeline."""
    scores = get_construct_scores(doc.text)
    threatened = infer_threatened_constructs(doc.text)
    phi = compute_phi(scores)

    return {
        "title": doc.title,
        "url": doc.url,
        "source_portal": doc.source_portal,
        "construct_scores": scores,
        "threatened_constructs": list(threatened),
        "phi": phi,
        "text_length": len(doc.text),
        "has_text": bool(doc.text.strip()),
    }


def process_for_training(doc: FOIADocument) -> dict[str, Any]:
    """Process a document for the training data pipeline."""
    scores = get_construct_scores(doc.text)

    return {
        "text": doc.text,
        "scores": scores,
        "metadata": {
            "source": doc.source_portal,
            "title": doc.title,
            "url": doc.url,
            "collection": doc.collection,
        },
    }


def run_dual_pipeline(documents: list[FOIADocument]) -> dict[str, list]:
    """Run both evidence and training pipelines on a list of documents."""
    evidence_results = []
    training_results = []

    for doc in documents:
        if not doc.text or not doc.text.strip():
            _logger.debug(f"Skipping empty document: {doc.title}")
            continue

        try:
            evidence_results.append(process_for_evidence(doc))
            training_results.append(process_for_training(doc))
        except Exception as e:
            _logger.warning(f"Failed to process {doc.title}: {e}")

    _logger.info(f"Dual pipeline: {len(evidence_results)} evidence, {len(training_results)} training")
    return {"evidence": evidence_results, "training": training_results}
