"""Hypothesis evolution engine."""

from dataclasses import dataclass, replace
from datetime import datetime
import uuid


@dataclass(frozen=True)
class Hypothesis:
    """Immutable hypothesis object."""

    id: str
    text: str
    confidence: float
    timestamp: datetime
    parent_id: str | None = None

    # NEW: Welfare grounding fields
    welfare_relevance: float = 0.0  # [0, 1] score from welfare_scoring
    threatened_constructs: tuple[str, ...] = ()  # e.g., ("c", "lam")

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Hypothesis.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
        if not (0.0 <= self.welfare_relevance <= 1.0):
            raise ValueError(
                f"Hypothesis.welfare_relevance must be in [0.0, 1.0], got {self.welfare_relevance!r}"
            )

    @classmethod
    def create(cls, text: str, confidence: float):
        """Create new hypothesis."""
        return cls(
            id=str(uuid.uuid4()),
            text=text,
            confidence=confidence,
            timestamp=datetime.now(),
            parent_id=None
        )

    def update_confidence(self, new_confidence: float):
        """Spawn updated hypothesis."""
        return replace(
            self,
            id=str(uuid.uuid4()),
            confidence=new_confidence,
            timestamp=datetime.now(),
            parent_id=self.id
        )

    def combined_score(self, alpha: float = 0.7, beta: float = 0.3) -> float:
        """
        Weighted combination of epistemic confidence and welfare relevance.

        alpha > beta ensures epistemic honesty remains primary (Constitution Principle 1).
        Default: α=0.7, β=0.3 (epistemic confidence is >2× as important as welfare).

        Args:
            alpha: Weight for epistemic confidence
            beta: Weight for welfare relevance

        Returns:
            Combined score in [0, 1]
        """
        return alpha * self.confidence + beta * self.welfare_relevance
