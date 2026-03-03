"""3-layer entity filter for epstein-docs ingestion (ADR-016).

Removes noise entities that pollute the knowledge graph:

- **Layer 1 (junk):** FOIA redaction codes, emails, short strings, numeric refs
- **Layer 2 (fuzzy dedup):** Variant spellings → canonical form pre-pass
- **Layer 3 (role descriptions):** Possessives, inmate patterns, anonymized IDs

Dropped entities are logged to JSONL for later investigative review.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drop log — append-only JSONL writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityDrop:
    """Record of a single entity filtered during ingestion."""

    entity: str
    reason: str
    category: str  # "layer1" or "layer3"


class DropLog:
    """Append-only JSONL log of dropped entities."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._count = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, drop: EntityDrop) -> None:
        """Append a single drop record as a JSON line."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(drop)) + "\n")
        self._count += 1

    @property
    def count(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Layer 1 — junk entity detection
# ---------------------------------------------------------------------------

_FOIA_CODE_START = re.compile(r"^\(?b\)\(\d")
_FOIA_CODE_EMBEDDED = re.compile(r"\(b\)\(\d\)")
_BRACKET_REDACTED = re.compile(r"\[redact", re.IGNORECASE)
_PURE_NUMERIC = re.compile(r"^\d+$")


def is_junk(entity: str) -> str | None:
    """Return a reason string if *entity* is noise, else ``None``."""
    if len(entity) <= 2:
        return "too_short"
    if _FOIA_CODE_START.search(entity):
        return "foia_code"
    if _FOIA_CODE_EMBEDDED.search(entity):
        return "foia_code_embedded"
    if _BRACKET_REDACTED.search(entity):
        return "bracket_redacted"
    if "@" in entity:
        return "email_or_handle"
    if _PURE_NUMERIC.match(entity):
        return "numeric_ref"
    return None


# ---------------------------------------------------------------------------
# Layer 3 — role description detection
# ---------------------------------------------------------------------------

_POSSESSIVE = re.compile(r"['\u2019]s\b")
_INMATE_PREFIX = re.compile(r"^inmate\s", re.IGNORECASE)
_DOT_ANONYMIZED = re.compile(r"^\.[A-Z]")
_ANONYMIZED_OFFICER = re.compile(r"\*[A-Z]|\bOFFCR\b|\bCBP\b")
_ALL_LOWERCASE = re.compile(r"^[a-z][a-z\s-]+$")

# Role prefixes that indicate a generic description, NOT a real name.
# Excludes titles that precede real names (Agent, Judge, Officer, etc.)
_ROLE_PREFIXES: frozenset[str] = frozenset({
    "defendant",
    "defendants",
    "co-conspirator",
    "co-conspirators",
    "conspirator",
    "conspirators",
    "prosecutor",
    "prosecutors",
    "plaintiff",
    "plaintiffs",
    "petitioner",
    "petitioners",
    "respondent",
    "respondents",
    "complainant",
    "complainants",
    "witness",
    "witnesses",
    "victim",
    "victims",
    "detainee",
    "detainees",
    "informant",
    "informants",
    "suspect",
    "suspects",
    "fugitive",
    "fugitives",
    "counsel",
    "attorney",
    "attorneys",
    "the government",
    "the defendant",
    "unknown",
    "unidentified",
    "john doe",
    "jane doe",
})


def is_role_description(entity: str) -> str | None:
    """Return a reason string if *entity* is a role description, else ``None``."""
    if _POSSESSIVE.search(entity):
        return "possessive_phrase"
    if _INMATE_PREFIX.search(entity):
        return "inmate_pattern"
    if _DOT_ANONYMIZED.match(entity):
        return "dot_anonymized"
    if _ANONYMIZED_OFFICER.search(entity):
        return "anonymized_officer"

    lower = entity.lower().strip()
    if lower in _ROLE_PREFIXES:
        return "role_prefix"

    # All-lowercase, >5 chars — likely informal/usernames, not real names
    if len(entity) > 5 and _ALL_LOWERCASE.match(entity):
        return "lowercase_informal"

    return None


# ---------------------------------------------------------------------------
# Layer 2 — fuzzy deduplication pre-pass
# ---------------------------------------------------------------------------


def _name_key(name: str) -> str:
    """Normalize a name for fuzzy comparison.

    1. Strip parentheticals
    2. Flip ``LASTNAME, FIRST`` → ``FIRST LASTNAME``
    3. Lowercase, strip non-alpha (except spaces), collapse whitespace
    """
    # Strip parentheticals
    name = re.sub(r"\s*\(.*?\)\s*", " ", name)

    # Flip LASTNAME, FIRST → FIRST LASTNAME
    if "," in name:
        parts = name.split(",", 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"

    # Lowercase, keep only alpha + spaces
    name = re.sub(r"[^a-zA-Z\s]", "", name).lower()
    return re.sub(r"\s+", " ", name).strip()


def build_fuzzy_mappings(
    entities: list[str],
    existing_mappings: dict[str, str],
    threshold: float = 0.75,
) -> dict[str, str]:
    """Build new ``{variant: canonical}`` mappings via fuzzy string matching.

    Compares all entity pairs using ``difflib.SequenceMatcher`` and groups
    similar names under a single canonical form.  Canonical selection:
    prefer existing mapping value > longer name > alphabetical.

    Entities already present in *existing_mappings* (as keys) are skipped.
    """
    new_mappings: dict[str, str] = {}

    # Filter to entities not already mapped
    candidates = [e for e in entities if e not in existing_mappings]
    if not candidates:
        return new_mappings

    # Build name keys for comparison
    keys = {e: _name_key(e) for e in candidates}

    # Group by canonical form using union-find style grouping
    canonical_for: dict[str, str] = {}  # name_key → canonical entity

    for i, entity_a in enumerate(candidates):
        key_a = keys[entity_a]
        if not key_a:
            continue

        for entity_b in candidates[i + 1 :]:
            key_b = keys[entity_b]
            if not key_b:
                continue

            ratio = SequenceMatcher(None, key_a, key_b).ratio()
            if ratio >= threshold:
                # Determine canonical: existing mapping > longer > alphabetical
                canon_a = canonical_for.get(key_a)
                canon_b = canonical_for.get(key_b)

                # Collect all variants for this group
                group = {entity_a, entity_b}
                if canon_a:
                    group.add(canon_a)
                if canon_b:
                    group.add(canon_b)

                # Pick canonical: prefer one already in existing_mappings values
                existing_values = set(existing_mappings.values())
                canonical = None
                for candidate in group:
                    if candidate in existing_values:
                        canonical = candidate
                        break

                if canonical is None:
                    # Prefer longer name, then alphabetical
                    canonical = sorted(group, key=lambda n: (-len(n), n))[0]

                canonical_for[key_a] = canonical
                canonical_for[key_b] = canonical

    # Build variant→canonical mappings
    for entity in candidates:
        key = keys[entity]
        if key and key in canonical_for:
            canon = canonical_for[key]
            if entity != canon:
                new_mappings[entity] = canon

    return new_mappings


# ---------------------------------------------------------------------------
# Orchestrator — apply Layer 1 + Layer 3 filters
# ---------------------------------------------------------------------------


def filter_entities(
    entities: list[str],
    drop_log: DropLog | None = None,
) -> list[str]:
    """Apply Layer 1 (junk) and Layer 3 (role description) filters.

    Returns a list of clean entities.  Logs drops to *drop_log* if provided.
    """
    clean: list[str] = []
    for entity in entities:
        # Layer 1: junk
        reason = is_junk(entity)
        if reason is not None:
            if drop_log is not None:
                drop_log.record(EntityDrop(entity=entity, reason=reason, category="layer1"))
            continue

        # Layer 3: role description
        reason = is_role_description(entity)
        if reason is not None:
            if drop_log is not None:
                drop_log.record(EntityDrop(entity=entity, reason=reason, category="layer3"))
            continue

        clean.append(entity)
    return clean
