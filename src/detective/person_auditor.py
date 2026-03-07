"""Person auditor: claim decomposition, verification, and severity scoring.

Decomposes compound claims about a person into atomic claims, verifies each
against available evidence, and computes an overall severity score. Uses LLM
for decomposition/verification when a provider is available, falls back to
heuristics otherwise.

See ADR-027 for design rationale.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.core.scoring import parse_score

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLAIM_LINE_RE: re.Pattern[str] = re.compile(r"^\s*claim:\s*(.+)", re.MULTILINE | re.IGNORECASE)
_STATUS_RE: re.Pattern[str] = re.compile(
    r"status:\s*(supported|contradicted|unverified|partially_supported)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+")

# Claim type keyword patterns
_TEMPORAL_RE: re.Pattern[str] = re.compile(
    r"\b(\d{4}|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|between\s+\d|during\s+the)\b",
    re.IGNORECASE,
)
_FINANCIAL_RE: re.Pattern[str] = re.compile(
    r"\b(\$|fund(?:s|ing)?|payment|transfer|account|financ|money|wire|bank)\b",
    re.IGNORECASE,
)
_LOCATION_RE: re.Pattern[str] = re.compile(
    r"\b(traveled|visited|located|island|flew|airport|address|residence)\b",
    re.IGNORECASE,
)
_ASSOCIATION_RE: re.Pattern[str] = re.compile(
    r"\b(associated|met\s+with|connection|relationship|contact|knew|introduced)\b",
    re.IGNORECASE,
)
_ROLE_RE: re.Pattern[str] = re.compile(
    r"\b(served\s+as|employed|director|manager|officer|role|position|title)\b",
    re.IGNORECASE,
)

# Severity formula weights
_SEVERITY_CONTRADICTION_WEIGHT: float = 0.40
_SEVERITY_CONSTRUCT_WEIGHT: float = 0.35
_SEVERITY_VOLUME_WEIGHT: float = 0.25
_SEVERITY_VOLUME_NORMALIZER: int = 20  # 20+ verifications = max volume score
_WELFARE_CONSTRUCT_COUNT: int = 8  # total constructs in Phi formula

# Keyword overlap thresholds for heuristic verification
_OVERLAP_SUPPORTED_THRESHOLD: float = 0.4
_OVERLAP_PARTIAL_THRESHOLD: float = 0.2


# ---------------------------------------------------------------------------
# Provider Protocol (subset of ModelProvider)
# ---------------------------------------------------------------------------

@runtime_checkable
class _Provider(Protocol):
    def complete(self, prompt: str, **kwargs: object) -> str: ...


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomicClaim:
    """A single, verifiable claim extracted from a compound statement."""

    text: str
    source_text: str
    claim_type: str  # temporal, association, financial, location, role, action, other
    person: str


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying a single claim against evidence."""

    claim: AtomicClaim
    status: str  # supported, contradicted, unverified, partially_supported
    confidence: float
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )


@dataclass(frozen=True)
class PersonAudit:
    """Complete audit of a person's claims."""

    person: str
    claims: tuple[AtomicClaim, ...]
    verifications: tuple[VerificationResult, ...]
    overall_confidence: float
    contradiction_count: int
    severity_score: float

    @property
    def supported_count(self) -> int:
        return sum(1 for v in self.verifications if v.status == "supported")

    @property
    def contradicted_count(self) -> int:
        return sum(1 for v in self.verifications if v.status == "contradicted")

    @property
    def unverified_count(self) -> int:
        return sum(1 for v in self.verifications if v.status == "unverified")


# ---------------------------------------------------------------------------
# Claim type classification (heuristic)
# ---------------------------------------------------------------------------


def _classify_claim_type(text: str) -> str:
    """Classify a claim into a type using keyword matching."""
    if _TEMPORAL_RE.search(text):
        return "temporal"
    if _FINANCIAL_RE.search(text):
        return "financial"
    if _LOCATION_RE.search(text):
        return "location"
    if _ASSOCIATION_RE.search(text):
        return "association"
    if _ROLE_RE.search(text):
        return "role"
    return "other"


# ---------------------------------------------------------------------------
# Decompose claims
# ---------------------------------------------------------------------------


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences, filtering short fragments."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def decompose_claims(
    person: str,
    text: str,
    provider: _Provider | None = None,
) -> tuple[AtomicClaim, ...]:
    """Decompose compound text into atomic claims about a person.

    Uses LLM when provider is available, falls back to sentence splitting.
    """
    if not text or not text.strip():
        return ()

    claims: list[AtomicClaim] = []

    if provider is not None:
        prompt = (
            f"Decompose the following text into atomic, verifiable claims "
            f"about {person}. Output each claim on its own line prefixed "
            f"with 'claim:'. Only include claims directly about {person}.\n\n"
            f"Text: {text}"
        )
        response = provider.complete(prompt)
        matches = _CLAIM_LINE_RE.findall(response)

        if matches:
            for match in matches:
                claim_text = match.strip()
                if claim_text:
                    claims.append(AtomicClaim(
                        text=claim_text,
                        source_text=text,
                        claim_type=_classify_claim_type(claim_text),
                        person=person,
                    ))
            return tuple(claims)
        # LLM didn't return parseable claims — fall through to heuristic

    # Heuristic fallback: sentence splitting
    sentences = _sentence_split(text)
    for sentence in sentences:
        claims.append(AtomicClaim(
            text=sentence,
            source_text=text,
            claim_type=_classify_claim_type(sentence),
            person=person,
        ))

    return tuple(claims)


# ---------------------------------------------------------------------------
# Verify claim
# ---------------------------------------------------------------------------


def _keyword_overlap(claim_text: str, evidence_text: str) -> float:
    """Compute keyword overlap between claim and evidence."""
    claim_words = set(claim_text.lower().split())
    evidence_words = set(evidence_text.lower().split())
    # Remove very common words
    stop = {"the", "a", "an", "in", "on", "at", "to", "of", "and", "or", "was", "is", "with"}
    claim_words -= stop
    evidence_words -= stop
    if not claim_words:
        return 0.0
    return len(claim_words & evidence_words) / len(claim_words)


def verify_claim(
    claim: AtomicClaim,
    evidence_texts: list[str],
    provider: _Provider | None = None,
) -> VerificationResult:
    """Verify a claim against evidence texts.

    Uses LLM when provider is available, falls back to keyword overlap.
    """
    if not evidence_texts:
        return VerificationResult(
            claim=claim,
            status="unverified",
            confidence=0.0,
            supporting_evidence=(),
            contradicting_evidence=(),
        )

    if provider is not None:
        evidence_block = "\n".join(
            f"[{i+1}] {e[:500]}" for i, e in enumerate(evidence_texts[:5])
        )
        prompt = (
            f"Verify the following claim against the evidence provided.\n\n"
            f"Claim: {claim.text}\n"
            f"Person: {claim.person}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            f"Respond with:\n"
            f"status: <supported|contradicted|unverified|partially_supported>\n"
            f"confidence: <0.0 to 1.0>"
        )
        response = provider.complete(prompt)

        status_match = _STATUS_RE.search(response)
        status = status_match.group(1).lower() if status_match else "unverified"
        confidence = parse_score(response, default=0.5)

        supporting = tuple(
            e[:200] for e in evidence_texts if _keyword_overlap(claim.text, e) > _OVERLAP_PARTIAL_THRESHOLD
        ) if status in ("supported", "partially_supported") else ()
        contradicting = tuple(
            e[:200] for e in evidence_texts if _keyword_overlap(claim.text, e) > _OVERLAP_PARTIAL_THRESHOLD
        ) if status == "contradicted" else ()

        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
        )

    # Heuristic fallback: keyword overlap
    best_overlap = 0.0
    supporting: list[str] = []
    for evidence in evidence_texts:
        overlap = _keyword_overlap(claim.text, evidence)
        if overlap > best_overlap:
            best_overlap = overlap
        if overlap > _OVERLAP_PARTIAL_THRESHOLD:
            supporting.append(evidence[:200])

    if best_overlap >= _OVERLAP_SUPPORTED_THRESHOLD:
        status = "supported"
    elif best_overlap >= _OVERLAP_PARTIAL_THRESHOLD:
        status = "partially_supported"
    else:
        status = "unverified"

    return VerificationResult(
        claim=claim,
        status=status,
        confidence=min(1.0, best_overlap),
        supporting_evidence=tuple(supporting),
        contradicting_evidence=(),
    )


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------


def compute_severity(
    verification_count: int,
    contradiction_count: int,
    threatened_constructs: tuple[str, ...] = (),
) -> float:
    """Compute severity score for a person audit.

    Formula: 40% contradiction ratio + 35% welfare construct coverage + 25% volume.
    """
    if verification_count == 0:
        return 0.0

    contradiction_ratio = contradiction_count / verification_count
    construct_coverage = len(threatened_constructs) / _WELFARE_CONSTRUCT_COUNT
    volume_score = min(1.0, verification_count / _SEVERITY_VOLUME_NORMALIZER)

    raw = (
        _SEVERITY_CONTRADICTION_WEIGHT * contradiction_ratio
        + _SEVERITY_CONSTRUCT_WEIGHT * construct_coverage
        + _SEVERITY_VOLUME_WEIGHT * volume_score
    )
    return min(1.0, max(0.0, raw))


# ---------------------------------------------------------------------------
# Full audit pipeline
# ---------------------------------------------------------------------------


def audit_person(
    person: str,
    finding_texts: list[str],
    evidence_texts: list[str],
    provider: _Provider | None = None,
    threatened_constructs: tuple[str, ...] = (),
) -> PersonAudit:
    """Full audit: decompose claims, verify each, aggregate, score severity."""
    if not finding_texts:
        return PersonAudit(
            person=person,
            claims=(),
            verifications=(),
            overall_confidence=0.0,
            contradiction_count=0,
            severity_score=0.0,
        )

    # Decompose all finding texts into atomic claims
    all_claims: list[AtomicClaim] = []
    for text in finding_texts:
        claims = decompose_claims(person, text, provider=provider)
        all_claims.extend(claims)

    # Verify each claim
    verifications: list[VerificationResult] = []
    for claim in all_claims:
        result = verify_claim(claim, evidence_texts, provider=provider)
        verifications.append(result)

    # Aggregate
    contradiction_count = sum(1 for v in verifications if v.status == "contradicted")
    confidences = [v.confidence for v in verifications if v.confidence > 0]
    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    severity = compute_severity(
        verification_count=len(verifications),
        contradiction_count=contradiction_count,
        threatened_constructs=threatened_constructs,
    )

    return PersonAudit(
        person=person,
        claims=tuple(all_claims),
        verifications=tuple(verifications),
        overall_confidence=overall_confidence,
        contradiction_count=contradiction_count,
        severity_score=severity,
    )
