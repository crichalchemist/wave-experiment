from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.core.types import AssumptionType

_CONFIDENCE_THRESHOLD: float = 0.7
_MODEL_NAME: str = "distilbert-base-uncased"  # fine-tuned for bias detection


@dataclass(frozen=True)
class BiasDetection:
    """A detected cognitive bias with its type, score, and the text span that triggered it."""
    assumption_type: AssumptionType
    score: float
    span: str


@lru_cache(maxsize=1)
def _get_classifier():
    """Load DistilBERT classifier once. Cached after first call."""
    from transformers import pipeline
    return pipeline("text-classification", model=_MODEL_NAME, top_k=None)


def detect_cognitive_biases(text: str) -> list[BiasDetection]:
    """Run bias classifier and return detections above confidence threshold."""
    classifier = _get_classifier()
    results = classifier(text)
    return [
        BiasDetection(
            assumption_type=AssumptionType(r["label"]),
            score=r["score"],
            span=text,
        )
        for r in results
        if r["score"] >= _CONFIDENCE_THRESHOLD
    ]
