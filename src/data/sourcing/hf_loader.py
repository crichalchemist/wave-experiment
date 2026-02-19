"""
HuggingFace dataset loader for constitutional training data.

Loads legal, court, and regulatory text from public HF datasets.
Returns normalized dicts with text, source, and metadata preserved —
provenance tracking is required for all training examples per the
constitution's standpoint transparency principle.
"""
from __future__ import annotations

from typing import Any


def load_hf_legal_batch(
    dataset_name: str,
    split: str = "train",
    max_examples: int = 200,
    text_field: str = "text",
    config_name: str | None = None,
    keyword_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load a batch from a HuggingFace dataset, normalizing to {text, source, metadata}.

    Args:
        dataset_name: HF dataset identifier (e.g., "pile-of-law/pile-of-law").
        split: Dataset split ("train", "test", "validation").
        max_examples: Maximum number of examples to return.
        text_field: Name of the field containing the document text.
        config_name: Optional dataset configuration name.
        keyword_filter: If provided, only include examples containing this keyword.

    Returns:
        List of dicts: {text: str, source: str, metadata: dict}
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config_name:
        kwargs["name"] = config_name

    ds = load_dataset(dataset_name, **kwargs)

    results: list[dict[str, Any]] = []
    for example in ds:
        if len(results) >= max_examples:
            break
        text = example.get(text_field, "")
        if not text or not isinstance(text, str):
            continue
        if keyword_filter and keyword_filter.lower() not in text.lower():
            continue
        results.append({
            "text": text,
            "source": f"huggingface:{dataset_name}",
            "metadata": {
                "dataset": dataset_name,
                "split": split,
                "config": config_name,
                "original_keys": list(example.keys()),
            },
        })

    return results
