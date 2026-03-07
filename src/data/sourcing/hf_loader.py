"""
HuggingFace dataset loader for constitutional training data.

Loads legal, court, and regulatory text from public HF datasets.
Returns SourceDocument instances with provenance metadata.
"""
from __future__ import annotations

import logging
from typing import Any

from src.data.sourcing.types import SourceDocument, limit_results

_logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore[assignment]


def load_hf_legal_batch(
    *,
    dataset_name: str,
    split: str = "train",
    max_documents: int = 200,
    text_field: str = "text",
    config_name: str | None = None,
    keyword_filter: str | None = None,
) -> list[SourceDocument]:
    """
    Load a batch from a HuggingFace dataset, returning SourceDocument instances.
    """
    if load_dataset is None:
        raise ImportError("pip install datasets")

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config_name:
        kwargs["name"] = config_name

    ds = load_dataset(dataset_name, **kwargs)

    results: list[SourceDocument] = []
    for example in ds:
        if len(results) >= max_documents:
            break
        text = example.get(text_field, "")
        if not text or not isinstance(text, str):
            continue
        if keyword_filter and keyword_filter.lower() not in text.lower():
            continue
        results.append(SourceDocument(
            text=text,
            source=f"huggingface:{dataset_name}",
            metadata={
                "dataset": dataset_name,
                "split": split,
                "config": config_name,
                "original_keys": list(example.keys()),
            },
        ))

    return limit_results(results, max_documents)
