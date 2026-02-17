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

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Hypothesis.confidence must be in [0.0, 1.0], got {self.confidence!r}"
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
