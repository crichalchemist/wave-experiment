"""
Module C: Geopolitical Presumption Detector.

Detects language that assumes institutional actors behaved according to their
stated mandates without questioning whether their actual interests aligned —
the third failure mode in docs/constitution.md.

Example: "The regulator properly reviewed all disclosures" presumes the
regulator's actual behavior matched its formal function. In influence network
analysis, this assumption often masks the most significant gaps: when powerful
actors exploit institutional legitimacy as cover for behavior that contradicts
that legitimacy.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.core.providers import ModelProvider
from src.core.types import AssumptionType
from src.core.scoring import parse_score as _parse_score

_logger = logging.getLogger(__name__)

# Institutional actor patterns — entities whose "normal" behavior is often presumed
_ACTOR_PATTERNS: tuple[str, ...] = (
    r"\b(?:the\s+)?(?:SEC|DOJ|FBI|CIA|NSA|DOD|DOE|DEA|ATF)\b",
    r"\b(?:the\s+)?(?:regulator|regulatory\s+body|oversight\s+committee)\b",
    r"\b(?:the\s+)?(?:prosecutor|attorney\s+general|district\s+attorney)\b",
    r"\b(?:the\s+)?(?:intelligence\s+(?:agency|service|community))\b",
    r"\b(?:the\s+)?(?:government|administration|authorities)\b",
    r"\b(?:the\s+)?(?:court|judiciary|judge)\b",
    r"\b(?:the\s+)?(?:embassy|consulate|diplomatic\s+mission)\b",
    r"\b(?:the\s+)?(?:ministry|minister|secretary)\b",
    r"\b(?:the\s+)?(?:Interpol|Europol|FATF)\b",
)

# Presumption verbs — language that assumes institutional behavior was normative
_PRESUMPTION_VERBS: tuple[str, ...] = (
    r"\bproperly\s+(?:reviewed|investigated|monitored|approved|enforced)\b",
    r"\bfully\s+cooperated\b",
    r"\bper\s+standard\b",
    r"\bper\s+(?:standard\s+)?(?:protocol|procedure|mandate|regulation|law)\b",
    r"\bas\s+(?:required|mandated|stipulated)\s+by\b",
    r"\bin\s+accordance\s+with\b",
    r"\bas\s+expected\b",
    r"\bfollowed\s+standard\b",
    r"\bdeclined\s+to\s+pursue\s+(?:charges\s+)?(?:based\s+on|due\s+to)\b",
    r"\binsufficient\s+evidence\b",
    r"\bdetermined\s+that\s+no\b",
    r"\bfound\s+no\s+(?:evidence|wrongdoing|violation)\b",
    r"\bper\s+(?:their|its)\s+mandate\b",
    r"\bthorough\s+investigation\b",
)

_SCORE_THRESHOLD: float = 0.5

_SCORE_PROMPT = (
    "You are evaluating whether the following sentence contains a 'geopolitical presumption' — "
    "an unstated assumption that an institutional actor behaved according to its stated mandate, "
    "without questioning whether its actual interests or external pressures shaped its behavior "
    "differently.\n\n"
    "This is especially significant when powerful actors are described as having 'properly' "
    "followed procedure in contexts where that procedure may have served to suppress inquiry.\n\n"
    "Rate from 0.0 (no presumption) to 1.0 (strong presumption).\n"
    "Reply with ONLY: score: <float>\n\n"
    "Sentence: {sentence}\n"
    "Identified actor: {actor}"
)


@dataclass(frozen=True)
class GeopoliticalDetection:
    """
    A detected instance of geopolitical presumption language.

    Frozen for the same reason as all domain objects: detections form an
    immutable audit trail. A detection that changes after the fact is not
    a detection — it is a revision of the record.
    """
    assumption_type: AssumptionType
    score: float            # 0.0–1.0
    source_text: str        # sentence containing the presumption
    presumed_actor: str     # the institutional actor whose behavior is presumed normative

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")


def detect_geopolitical_presumptions(
    text: str,
    provider: ModelProvider,
    threshold: float = _SCORE_THRESHOLD,
) -> list[GeopoliticalDetection]:
    """
    Scan text for geopolitical presumption language.

    A sentence must contain both an institutional actor AND a presumption verb
    to trigger scoring — reducing false positives from actor mentions alone.
    Multiple actor matches within one sentence each produce an independent
    detection (e.g. "The SEC and DOJ jointly reviewed..." yields two).

    Args:
        text: Document text to analyze.
        provider: LLM provider for context-sensitive scoring.
        threshold: Minimum score for inclusion (default 0.5).

    Returns:
        List of GeopoliticalDetection instances, ordered by score descending.
    """
    detections: list[GeopoliticalDetection] = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        # Collect all actor matches in this sentence (finditer catches multiple)
        all_actor_matches: list[re.Match] = []  # type: ignore[type-arg]
        for actor_pat in _ACTOR_PATTERNS:
            for m in re.finditer(actor_pat, sentence, re.IGNORECASE):
                all_actor_matches.append(m)

        if not all_actor_matches:
            continue

        has_presumption_verb = any(
            re.search(vp, sentence, re.IGNORECASE) for vp in _PRESUMPTION_VERBS
        )
        if not has_presumption_verb:
            continue

        # Score once per unique actor match
        seen_actors: set[str] = set()
        for actor_match in all_actor_matches:
            actor_text = actor_match.group(0).strip()
            # Deduplicate by normalised actor text to avoid double-counting
            actor_key = actor_text.lower()
            if actor_key in seen_actors:
                continue
            seen_actors.add(actor_key)

            prompt = _SCORE_PROMPT.format(sentence=sentence, actor=actor_text)
            raw = provider.complete(prompt)
            score = _parse_score(raw)

            if score >= threshold:
                detections.append(GeopoliticalDetection(
                    assumption_type=AssumptionType.GEOPOLITICAL_PRESUMPTION,
                    score=score,
                    source_text=sentence,
                    presumed_actor=actor_text,
                ))

    return sorted(detections, key=lambda d: d.score, reverse=True)
