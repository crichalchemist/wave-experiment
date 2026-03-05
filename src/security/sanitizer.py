from __future__ import annotations
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

_logger = logging.getLogger(__name__)

# Injection pattern signatures
_INSTRUCTION_OVERRIDE_PATTERN = re.compile(
    r"(?i)(ignore|disregard|forget|override|bypass)\s+(previous|prior|above|all)?\s*(instructions?|rules?|guidelines?|constitution|constraints?)",
)
_ROLE_SWITCH_PATTERN = re.compile(
    r"(?i)(you are now|act as|pretend (you are|to be)|roleplay as|your (new )?role is)",
)
_FAKE_TURN_PATTERN = re.compile(
    r"(?m)^(SYSTEM|ASSISTANT|USER|HUMAN|AI|CLAUDE):\s*",
)
_CONSTITUTION_OVERRIDE_PATTERN = re.compile(
    r"(?i)(ignore|override|dismiss|disregard).{0,30}(constitution|moral compass|principles?|epistemic)",
)

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_INSTRUCTION_OVERRIDE_PATTERN, "instruction_override"),
    (_ROLE_SWITCH_PATTERN, "role_switch"),
    (_FAKE_TURN_PATTERN, "fake_conversation_turn"),
    (_CONSTITUTION_OVERRIDE_PATTERN, "constitution_override"),
]

RiskLevel = Literal["low", "medium", "high", "critical"]

_RISK_MAP: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}
_FINDING_RISK: dict[str, RiskLevel] = {
    "constitution_override": "critical",
    "instruction_override": "high",
    "role_switch": "high",
    "fake_conversation_turn": "medium",
    "unicode_control": "medium",
}


@dataclass(frozen=True)
class SanitizationResult:
    """
    Outcome of sanitizing a document.
    Injection findings are investigative data — a document designed to suppress
    gap detection is itself evidence of deliberate concealment.
    """
    safe_text: str
    injection_detected: bool
    risk_level: RiskLevel
    findings: tuple[str, ...]  # each entry names a detected pattern type


def _strip_unicode_controls(text: str) -> tuple[str, bool]:
    """Remove invisible/directional Unicode controls that can hide injection payloads."""
    cleaned = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cf", "Cc") or ch in ("\n", "\t", "\r")
    )
    return cleaned, cleaned != text


def _highest_risk(findings: list[str]) -> RiskLevel:
    """Constitution override attempts carry the highest risk — they target the epistemic foundation."""
    if not findings:
        return "low"
    levels: list[RiskLevel] = [_FINDING_RISK.get(f, "low") for f in findings]
    return max(levels, key=lambda r: _RISK_MAP[r])


def sanitize_document(text: str) -> SanitizationResult:
    """
    Sanitize web-sourced text before model ingestion.
    Injection patterns are logged as findings, not silently discarded —
    an attempt to suppress gaps is itself a gap.
    """
    findings: list[str] = []

    clean_text, had_unicode = _strip_unicode_controls(text)
    if had_unicode:
        findings.append("unicode_control")

    for pattern, name in _PATTERNS:
        if pattern.search(clean_text):
            findings.append(name)

    return SanitizationResult(
        safe_text=clean_text,
        injection_detected=bool(findings),
        risk_level=_highest_risk(findings),
        findings=tuple(findings),
    )
