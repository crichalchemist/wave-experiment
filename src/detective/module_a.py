"""Cognitive Bias Detector (Module A)."""

from dataclasses import dataclass
from enum import Enum


class BiasType(Enum):
    CONFIRMATION = "confirmation_bias"
    ANCHORING = "anchoring"
    SURVIVORSHIP = "survivorship_bias"
    INGROUP = "ingroup_bias"


@dataclass
class BiasDetection:
    bias_type: BiasType
    location: str
    description: str
    confidence: float
    evidence: list[str]


def detect_cognitive_biases(text: str) -> list[BiasDetection]:
    """Detect cognitive biases in text."""
    # TODO: Implement bias detection
    return []
