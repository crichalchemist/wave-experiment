"""Reasoning trace capture for chain-of-thought scoring responses.

The DeepSeek-R1-Distill-Qwen-1.5B produces chain-of-thought reasoning before
scoring (e.g. "Okay, so I need to evaluate... [analysis] ... score: 0.6").
This module captures those traces as immutable records for persistence,
streaming, and display.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

# Unique phrases from each module's prompt template.
# Order matters: more specific patterns first to avoid false matches.
_MODULE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("module_a", re.compile(r"cognitive bias", re.IGNORECASE)),
    ("module_b", re.compile(r"historical determinism", re.IGNORECASE)),
    ("module_c", re.compile(r"geopolitical presumption", re.IGNORECASE)),
    ("evolution", re.compile(r"updated confidence", re.IGNORECASE)),
    ("graph", re.compile(r"plausibility of this relationship", re.IGNORECASE)),
    ("parallel_evolution", re.compile(r"evolving a hypothesis", re.IGNORECASE)),
)

# Extracts a float after "score:" or a bare float at the end of the response.
_SCORE_RE = re.compile(
    r"(?:score:\s*)(\d+(?:\.\d+)?)"
    r"|"
    r"(?:confidence:\s*)(\d+(?:\.\d+)?)"
    r"|"
    r"\b(\d\.\d+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class ReasoningTrace:
    """Immutable record of a single chain-of-thought scoring response."""

    id: str
    timestamp: str
    module: str
    prompt: str
    raw_response: str
    parsed_score: float | None
    model: str
    route: str
    duration_ms: int

    @staticmethod
    def create(
        *,
        prompt: str,
        raw_response: str,
        model: str,
        route: str,
        duration_ms: int,
    ) -> ReasoningTrace:
        """Factory that auto-generates id, timestamp, module, and parsed_score."""
        return ReasoningTrace(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            module=classify_module(prompt),
            prompt=prompt,
            raw_response=raw_response,
            parsed_score=try_parse_score(raw_response),
            model=model,
            route=route,
            duration_ms=duration_ms,
        )


def classify_module(prompt: str) -> str:
    """Identify which detective module generated a prompt via unique phrases."""
    for module_name, pattern in _MODULE_PATTERNS:
        if pattern.search(prompt):
            return module_name
    return "unknown"


def try_parse_score(response: str) -> float | None:
    """Extract a float score from a CoT response. Returns None if absent."""
    match = _SCORE_RE.search(response)
    if match is None:
        return None
    # Return the first non-None group
    for group in match.groups():
        if group is not None:
            value = float(group)
            return value
    return None
