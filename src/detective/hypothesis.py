"""Hypothesis evolution engine."""

import logging
from dataclasses import dataclass, replace
from datetime import datetime
import uuid

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hypothesis:
    """Immutable hypothesis object."""

    id: str
    text: str
    confidence: float
    timestamp: datetime
    parent_id: str | None = None

    # Welfare grounding fields
    welfare_relevance: float = 0.0  # [0, 1] score from welfare_scoring
    threatened_constructs: tuple[str, ...] = ()  # e.g., ("c", "lam_P")

    # Curiosity: love aimed at truth (lam_L x xi coupling)
    curiosity_relevance: float = 0.0  # [0, 1] — how strongly this hypothesis
                                       # sits at the love/truth intersection

    trajectory_urgency: float = 0.0  # [0, 1] — forecast-informed urgency

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Hypothesis.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
        if not (0.0 <= self.welfare_relevance <= 1.0):
            raise ValueError(
                f"Hypothesis.welfare_relevance must be in [0.0, 1.0], got {self.welfare_relevance!r}"
            )
        if not (0.0 <= self.curiosity_relevance <= 1.0):
            raise ValueError(
                f"Hypothesis.curiosity_relevance must be in [0.0, 1.0], got {self.curiosity_relevance!r}"
            )
        if not (0.0 <= self.trajectory_urgency <= 1.0):
            raise ValueError(
                f"Hypothesis.trajectory_urgency must be in [0.0, 1.0], got {self.trajectory_urgency!r}"
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

    def combined_score(
        self,
        alpha: float = 0.55,
        beta: float = 0.30,
        gamma: float = 0.15,
        delta: float = 0.0,
    ) -> float:
        """
        Weighted combination of epistemic confidence, welfare relevance,
        curiosity relevance, and trajectory urgency.

        alpha > beta > gamma ensures epistemic honesty remains primary
        (Constitution Principle 1), while curiosity can surface hunches
        that would otherwise be buried by low confidence.

        Default: alpha=0.55, beta=0.30, gamma=0.15, delta=0.0 (sum=1.0 when delta=0).
        Bridge mode: alpha=0.45, beta=0.25, gamma=0.15, delta=0.15 (sum=1.0).

        Args:
            alpha: Weight for epistemic confidence
            beta: Weight for welfare relevance
            gamma: Weight for curiosity relevance (love aimed at truth)
            delta: Weight for trajectory urgency (forecast-informed)

        Returns:
            Combined score in [0, 1]
        """
        return (
            alpha * self.confidence
            + beta * self.welfare_relevance
            + gamma * self.curiosity_relevance
            + delta * self.trajectory_urgency
        )


# Named scoring weight schemes (ADR-010)
WEIGHTS_DEFAULT = {"alpha": 0.55, "beta": 0.30, "gamma": 0.15, "delta": 0.0}
WEIGHTS_BRIDGE = {"alpha": 0.45, "beta": 0.25, "gamma": 0.15, "delta": 0.15}
