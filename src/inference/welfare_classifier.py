"""
Semantic welfare construct classifier using fine-tuned DistilBERT.

Replaces keyword-based detection with semantic understanding.
Falls back gracefully to zero scores when the model is not yet trained.
"""
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple
import logging

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

# Hub model ID (pushed by HF Jobs training)
HUB_MODEL_ID = "crichalchemist/welfare-constructs-distilbert"

# Local model path (relative to project root) — fallback for development/offline use
MODEL_PATH = Path("models/welfare-constructs-distilbert")

CONSTRUCT_NAMES = ["c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"]


@lru_cache(maxsize=1)
def _load_welfare_classifier():
    """
    Lazy-load fine-tuned DistilBERT (cached singleton).

    Tries loading from Hugging Face Hub first, falls back to local model path.
    Raises FileNotFoundError if neither source is available.
    """
    # Try Hub first
    try:
        logger.info(f"Loading welfare classifier from Hub: {HUB_MODEL_ID}...")
        return pipeline(
            "text-classification",
            model=HUB_MODEL_ID,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None,
        )
    except (OSError, Exception) as e:
        logger.debug(f"Hub loading failed: {e}")

    # Fallback to local
    config_file = MODEL_PATH / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"Welfare classifier not found at Hub ({HUB_MODEL_ID}) "
            f"or local ({MODEL_PATH}). "
            f"Train with scripts/train_welfare_classifier_hf_job.py."
        )

    logger.info(f"Loading welfare classifier from local: {MODEL_PATH}...")
    return pipeline(
        "text-classification",
        model=str(MODEL_PATH),
        device=0 if torch.cuda.is_available() else -1,
        top_k=None,
    )


def get_construct_scores(text: str) -> Dict[str, float]:
    """
    Get semantic welfare construct scores in [0, 1].

    Returns a dict mapping each of the 8 construct symbols to a float score.
    Falls back to zero scores if the model is not yet trained.

    Args:
        text: Input text to classify.

    Returns:
        Dict with keys {c, kappa, j, p, eps, lam_L, lam_P, xi}, values in [0, 1].
    """
    try:
        classifier = _load_welfare_classifier()
    except (FileNotFoundError, ValueError, OSError):
        _load_welfare_classifier.cache_clear()
        logger.warning(
            "Welfare classifier model not found or invalid; returning zero scores. "
            "Train the model with scripts/train_welfare_classifier.py."
        )
        return {construct: 0.0 for construct in CONSTRUCT_NAMES}

    truncated = text[:2048]
    results = classifier(truncated, truncation=True, max_length=512)

    scores_dict: Dict[str, float] = {}
    if results and isinstance(results, list) and len(results) > 0:
        predictions = results[0]
        for i, construct in enumerate(CONSTRUCT_NAMES):
            label_key = f"LABEL_{i}"
            for pred in predictions:
                if pred["label"] == label_key:
                    scores_dict[construct] = pred["score"]
                    break

    # Ensure all constructs have a score (default 0.0 if missing)
    for construct in CONSTRUCT_NAMES:
        if construct not in scores_dict:
            scores_dict[construct] = 0.0

    return scores_dict


def infer_threatened_constructs(
    text: str,
    threshold: float = 0.3,
) -> Tuple[str, ...]:
    """
    Infer threatened constructs via semantic analysis.

    Args:
        text: Input text to analyse.
        threshold: Minimum score to consider a construct threatened.

    Returns:
        Sorted tuple of construct symbols above the threshold.
    """
    scores = get_construct_scores(text)
    threatened = [
        construct
        for construct, score in scores.items()
        if score >= threshold
    ]
    return tuple(sorted(threatened))
