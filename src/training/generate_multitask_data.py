"""Synthetic multi-task training data generator.

Produces JSONL samples with gap_type and assumption_type labels
using regex triggers from Module A/B/C patterns plus template sentences.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

# --- Regex triggers (aligned with Module A/B/C patterns) ---

_COGNITIVE_BIAS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bconfirm(?:s|ed|ing)\b", re.IGNORECASE),
    re.compile(r"\banchored?\b", re.IGNORECASE),
    re.compile(r"\bsurvivor(?:ship)?\b", re.IGNORECASE),
    re.compile(r"\bin-?group\b", re.IGNORECASE),
    re.compile(r"\bbias(?:es|ed)?\b", re.IGNORECASE),
]

_HISTORICAL_DETERMINISM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\balways been\b", re.IGNORECASE),
    re.compile(r"\binevitabl[ey]\b", re.IGNORECASE),
    re.compile(r"\bchronological order\b", re.IGNORECASE),
    re.compile(r"\bhistorical record shows\b", re.IGNORECASE),
    re.compile(r"\bnaturally led to\b", re.IGNORECASE),
]

_GEOPOLITICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bproperly\s+(?:review|oversight|regulat)", re.IGNORECASE),
    re.compile(r"\bregulat(?:or|ory)\s+(?:oversight|review|compliance)", re.IGNORECASE),
    re.compile(r"\binstitution(?:al)?\s+(?:norm|standard|practice)", re.IGNORECASE),
    re.compile(r"\bofficial\s+channel", re.IGNORECASE),
    re.compile(r"\bstandard\s+protocol", re.IGNORECASE),
]

_GAP_TEMPORAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btemporal\s+gap\b", re.IGNORECASE),
    re.compile(r"\bmissing\s+(?:period|year|month|time)", re.IGNORECASE),
    re.compile(r"\bno\s+records?\s+(?:from|between|during)\b", re.IGNORECASE),
    re.compile(r"\bunexplained\s+(?:silence|absence|gap)\b", re.IGNORECASE),
]

_GAP_EVIDENTIAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bno\s+(?:documentation|evidence|proof|record)\b", re.IGNORECASE),
    re.compile(r"\bunsupported\s+claim\b", re.IGNORECASE),
    re.compile(
        r"\bwithout\s+(?:documentation|evidence|corroboration)\b", re.IGNORECASE
    ),
    re.compile(r"\bclaim(?:s|ed)?\s+without\b", re.IGNORECASE),
]

_GAP_CONTRADICTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcontradiction\b", re.IGNORECASE),
    re.compile(r"\bconflicting\s+(?:information|accounts|statements)\b", re.IGNORECASE),
    re.compile(r"\binconsisten(?:t|cy)\b", re.IGNORECASE),
]

_GAP_NORMATIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bshould\s+have\s+been\s+(?:document|report|record)", re.IGNORECASE),
    re.compile(r"\brequired\s+(?:disclosure|filing|report)\b", re.IGNORECASE),
    re.compile(r"\bnormative\s+gap\b", re.IGNORECASE),
]

_GAP_DOCTRINAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bdoctrinal\b", re.IGNORECASE),
    re.compile(r"\bunstated\s+(?:rule|policy|procedure)\b", re.IGNORECASE),
    re.compile(r"\bassumed\s+(?:to\s+)?appl(?:y|ied)\b", re.IGNORECASE),
]


def classify_text(text: str) -> tuple[str | None, str | None]:
    """Classify text using regex triggers.

    Returns:
        (gap_type | None, assumption_type | None)
    """
    gap_type: str | None = None
    assumption_type: str | None = None

    # Check assumption types
    for pat in _COGNITIVE_BIAS_PATTERNS:
        if pat.search(text):
            assumption_type = "cognitive_bias"
            break
    if assumption_type is None:
        for pat in _HISTORICAL_DETERMINISM_PATTERNS:
            if pat.search(text):
                assumption_type = "historical_determinism"
                break
    if assumption_type is None:
        for pat in _GEOPOLITICAL_PATTERNS:
            if pat.search(text):
                assumption_type = "geopolitical_presumption"
                break

    # Check gap types
    for pat in _GAP_TEMPORAL_PATTERNS:
        if pat.search(text):
            gap_type = "temporal"
            break
    if gap_type is None:
        for pat in _GAP_EVIDENTIAL_PATTERNS:
            if pat.search(text):
                gap_type = "evidential"
                break
    if gap_type is None:
        for pat in _GAP_CONTRADICTION_PATTERNS:
            if pat.search(text):
                gap_type = "contradiction"
                break
    if gap_type is None:
        for pat in _GAP_NORMATIVE_PATTERNS:
            if pat.search(text):
                gap_type = "normative"
                break
    if gap_type is None:
        for pat in _GAP_DOCTRINAL_PATTERNS:
            if pat.search(text):
                gap_type = "doctrinal"
                break

    return gap_type, assumption_type


# --- Templates for synthetic data generation ---

_TEMPLATES: list[str] = [
    # Temporal gaps
    "There is a temporal gap in the records from {year1} to {year2}.",
    "No records from the period between {year1} and {year2} have been found.",
    "An unexplained silence exists in the documentation during {year1}-{year2}.",
    "The missing period from {year1} to {year2} remains unaddressed.",
    # Evidential gaps
    "No documentation was found supporting this claim.",
    "The assertion was made without evidence or corroboration.",
    "This claim lacks documentation from any verified source.",
    "No evidence has been presented to support the allegation.",
    # Contradiction gaps
    "There is a contradiction between the two witness statements.",
    "Conflicting information was found in the financial records.",
    "The accounts are inconsistent with the physical evidence.",
    # Normative gaps
    "The required disclosure was never filed with the agency.",
    "This transaction should have been reported under federal guidelines.",
    "A normative gap exists where mandatory reporting was absent.",
    # Doctrinal gaps
    "The doctrinal basis for this policy was never established.",
    "An unstated rule appears to govern these proceedings.",
    "Standard operating procedures were assumed to apply without verification.",
    # Cognitive bias assumptions
    "The evidence confirmed our initial assessment of the subject.",
    "Results were anchored to the first estimate provided.",
    "Only survivorship data was considered in the analysis.",
    # Historical determinism assumptions
    "Records have always been maintained in chronological order.",
    "The outcome was seen as inevitable given the circumstances.",
    "History naturally led to this conclusion.",
    # Geopolitical presumptions
    "The regulator properly reviewed all disclosures before approval.",
    "Regulatory oversight ensured compliance with all standards.",
    "The institutional norms were followed throughout the process.",
    "Communication went through official channels as required.",
    # Neutral (no labels)
    "The meeting was held on Tuesday at the downtown office.",
    "Three witnesses provided testimony during the proceedings.",
    "The document was received on {year1}-03-15.",
    "A total of 47 pages were submitted as part of the filing.",
]


def generate_synthetic_samples(
    count: int,
    output_path: str,
    seed: int | None = None,
) -> int:
    """Generate synthetic multi-task training samples and write to JSONL.

    Returns the number of samples written.
    """
    rng = random.Random(seed)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with p.open("w", encoding="utf-8") as f:
        for _ in range(count):
            template = rng.choice(_TEMPLATES)
            text = template.format(
                year1=rng.randint(1990, 2015),
                year2=rng.randint(2016, 2025),
            )
            gap_type, assumption_type = classify_text(text)
            rec: dict[str, str | None] = {"text": text}
            if gap_type is not None:
                rec["gap_type"] = gap_type
            if assumption_type is not None:
                rec["assumption_type"] = assumption_type
            f.write(json.dumps(rec) + "\n")
            written += 1

    return written
