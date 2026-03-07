"""Named Entity Recognition pipeline with spaCy primary + heuristic fallback.

Extracts PERSON, ORG, GPE, DATE, LOC, NORP, EVENT entities from text.
Returns frozen NerResult dataclass — immutable for evidence provenance.

Backend selection (lazy, on first call):
  1. en_core_web_trf (transformer, highest accuracy)
  2. en_core_web_sm (statistical, lighter)
  3. heuristic fallback (no deps required)

See ADR-025 for design rationale.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import
# ---------------------------------------------------------------------------

try:
    import spacy as _spacy

    _HAS_SPACY = True
except ImportError:
    _spacy = None  # type: ignore[assignment]
    _HAS_SPACY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_LABELS: frozenset[str] = frozenset({
    "PERSON", "ORG", "GPE", "DATE", "LOC", "NORP", "EVENT",
})

# Common sentence-starting words that look like entities but aren't
_SENTENCE_START_SKIP: frozenset[str] = frozenset({
    "The", "This", "That", "These", "Those", "He", "She", "It", "They",
    "We", "You", "His", "Her", "Its", "Their", "Our", "Your", "My",
    "However", "Therefore", "Furthermore", "Moreover", "Nevertheless",
    "Meanwhile", "Although", "Because", "Since", "While", "When",
    "Where", "Which", "What", "Who", "How", "But", "And", "Also",
    "After", "Before", "During", "Until", "Unless", "Despite",
    "According", "Both", "Each", "Every", "Some", "Many", "Most",
    "Several", "Other", "Another", "Such", "Any", "All", "No",
    "Not", "Only", "Just", "Still", "Even", "Then", "Now", "Here",
    "There", "Very", "Much", "More", "Less", "Well", "Too",
})

# Minimum entity text length for heuristic extraction
_MIN_ENTITY_LENGTH: int = 3

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NerEntity:
    """A single named entity extracted from text."""

    text: str
    label: str  # PERSON, ORG, GPE, DATE, LOC, NORP, EVENT
    start: int
    end: int


@dataclass(frozen=True)
class NerResult:
    """Result of NER extraction — immutable evidence artifact."""

    entities: tuple[NerEntity, ...]
    backend: str  # "spacy" or "heuristic"
    text_length: int

    @property
    def persons(self) -> tuple[NerEntity, ...]:
        return tuple(e for e in self.entities if e.label == "PERSON")

    @property
    def organizations(self) -> tuple[NerEntity, ...]:
        return tuple(e for e in self.entities if e.label == "ORG")

    def unique_texts(self, label: str | None = None) -> tuple[str, ...]:
        """Deduplicated entity texts, optionally filtered by label."""
        seen: set[str] = set()
        result: list[str] = []
        for e in self.entities:
            if label is not None and e.label != label:
                continue
            if e.text not in seen:
                seen.add(e.text)
                result.append(e.text)
        return tuple(result)


# ---------------------------------------------------------------------------
# spaCy backend (lazy loading)
# ---------------------------------------------------------------------------

_nlp_cache: object | None = None
_nlp_loaded: bool = False


def _load_spacy() -> object | None:
    """Lazy-load spaCy model: trf → sm → None."""
    global _nlp_cache, _nlp_loaded
    if _nlp_loaded:
        return _nlp_cache

    _nlp_loaded = True

    if not _HAS_SPACY:
        _logger.info("spaCy not installed — using heuristic NER fallback")
        return None

    for model_name in ("en_core_web_trf", "en_core_web_sm"):
        try:
            _nlp_cache = _spacy.load(model_name)
            _logger.info("Loaded spaCy model: %s", model_name)
            return _nlp_cache
        except OSError:
            _logger.debug("spaCy model %s not found, trying next", model_name)

    _logger.info("No spaCy model found — using heuristic NER fallback")
    return None


def _spacy_extract(text: str, nlp: object) -> NerResult:
    """Extract entities using spaCy pipeline."""
    doc = nlp(text)  # type: ignore[operator]
    entities: list[NerEntity] = []
    for ent in doc.ents:  # type: ignore[attr-defined]
        if ent.label_ in _VALID_LABELS:
            entities.append(NerEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            ))
    return NerResult(
        entities=tuple(entities),
        backend="spacy",
        text_length=len(text),
    )


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

# Matches sequences of capitalized words (multi-word entity candidates)
_CAPITALIZED_SEQ_RE = re.compile(
    r"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b"
)


def _heuristic_extract(text: str) -> NerResult:
    """Extract entities using improved capitalization heuristic.

    Better than the old word-splitting approach:
    - Captures multi-word entities ("Jeffrey Epstein")
    - Skips common sentence starters
    - Filters short/junk tokens
    """
    entities: list[NerEntity] = []

    for match in _CAPITALIZED_SEQ_RE.finditer(text):
        candidate = match.group(1)
        offset = match.start()

        # Strip leading sentence-starter words from multi-word matches
        words = candidate.split()
        while words and words[0] in _SENTENCE_START_SKIP:
            stripped = words.pop(0)
            offset += len(stripped) + 1  # +1 for space
        candidate = " ".join(words)

        # Skip too-short tokens
        if len(candidate) < _MIN_ENTITY_LENGTH:
            continue

        # Classify heuristically: all-caps short → ORG, else PERSON
        label = "ORG" if candidate.isupper() and len(candidate) <= 6 else "PERSON"

        entities.append(NerEntity(
            text=candidate,
            label=label,
            start=offset,
            end=offset + len(candidate),
        ))

    return NerResult(
        entities=tuple(entities),
        backend="heuristic",
        text_length=len(text),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> NerResult:
    """Extract named entities from text.

    Uses spaCy if available (en_core_web_trf → en_core_web_sm),
    falls back to improved capitalization heuristic.
    """
    if not text:
        return NerResult(entities=(), backend="heuristic", text_length=0)

    nlp = _load_spacy()
    if nlp is not None:
        return _spacy_extract(text, nlp)
    return _heuristic_extract(text)
