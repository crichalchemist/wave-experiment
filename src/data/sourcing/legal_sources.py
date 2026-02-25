"""
Legal domain source configurations for written-vs-applied DPO training.

Each domain maps to HuggingFace dataset configs and keyword filters that
surface documents where statute-enforcement gaps are likely present.
"""
from __future__ import annotations

from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore[assignment]


LEGAL_DOMAIN_CONFIGS: dict[str, dict[str, Any]] = {
    "criminal_justice": {
        "hf_dataset": "pile-of-law/pile-of-law",
        "hf_config": "r_legaladvice",
        "keyword_filters": [
            "sentencing", "plea", "counsel", "miranda",
            "fourth amendment", "excessive force", "bail",
            "prosecutorial discretion", "mandatory minimum",
        ],
        "description": "Criminal justice: sentencing, policing, rights of the accused",
    },
    "territorial_rights": {
        "hf_dataset": "pile-of-law/pile-of-law",
        "hf_config": "r_legaladvice",
        "keyword_filters": [
            "territory", "puerto rico", "guam", "tribal",
            "insular", "samoa", "virgin islands",
            "territorial", "commonwealth", "sovereignty",
        ],
        "description": "Territorial rights: US territories, tribal nations, Insular Cases",
    },
    "foia_transparency": {
        "hf_dataset": "pile-of-law/pile-of-law",
        "hf_config": "r_legaladvice",
        "keyword_filters": [
            "foia", "freedom of information", "disclosure",
            "public records", "redact", "exemption",
            "glomar", "transparency", "withhold",
        ],
        "description": "FOIA/transparency: disclosure mandates vs. systematic denial",
    },
}


def load_legal_domain_batch(
    domain: str,
    max_examples: int = 200,
    split: str = "train",
) -> list[dict[str, Any]]:
    """
    Load documents for a specific legal domain from HuggingFace.

    Each returned dict has: text, source, metadata (with legal_domain tag).
    Streams the dataset and applies keyword filtering to find relevant docs.
    """
    if domain not in LEGAL_DOMAIN_CONFIGS:
        raise ValueError(f"Unknown legal domain: {domain}. Options: {list(LEGAL_DOMAIN_CONFIGS)}")

    if load_dataset is None:
        raise ImportError("pip install datasets")

    config = LEGAL_DOMAIN_CONFIGS[domain]
    ds = load_dataset(
        config["hf_dataset"],
        name=config["hf_config"],
        split=split,
        streaming=True,
    )

    keywords = [k.lower() for k in config["keyword_filters"]]
    results: list[dict[str, Any]] = []

    for example in ds:
        if len(results) >= max_examples:
            break
        text = example.get("text", "")
        if not text or not isinstance(text, str):
            continue
        text_lower = text.lower()
        if not any(kw in text_lower for kw in keywords):
            continue
        if len(text) < 200:
            continue
        results.append({
            "text": text,
            "source": f"huggingface:{config['hf_dataset']}",
            "metadata": {
                "dataset": config["hf_dataset"],
                "config": config["hf_config"],
                "legal_domain": domain,
                "matched_keywords": [kw for kw in keywords if kw in text_lower],
            },
        })

    return results
