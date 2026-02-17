from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.core.types import AssumptionType

_CONFIDENCE_THRESHOLD: float = 0.7
_MODEL_NAME: str = "distilbert-base-uncased"  # placeholder; replace with fine-tuned checkpoint when available


@dataclass(frozen=True)
class BiasDetection:
    """A detected cognitive bias; source_text is the full input passed to the classifier."""
    assumption_type: AssumptionType
    score: float
    source_text: str  # renamed from span — stores the full input text, not an extracted span

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"BiasDetection.score must be in [0.0, 1.0], got {self.score!r}")
        if not self.source_text:
            raise ValueError("BiasDetection.source_text must not be empty")


@lru_cache(maxsize=1)
def _get_classifier() -> Any:
    """Deferred import keeps module import fast; model loads only when first bias detection is requested."""
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
            source_text=text,
        )
        for r in results
        if r["score"] >= _CONFIDENCE_THRESHOLD
    ]
