"""MinHash/LSH near-duplicate detection for entities and documents.

Provides O(n) dedup via MinHash signatures + Locality-Sensitive Hashing.
Falls back to pairwise Jaccard on raw shingle sets when datasketch is not
installed, and delegates to difflib for small entity sets (< 500).

See ADR-026 for design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import
# ---------------------------------------------------------------------------

try:
    from datasketch import MinHash as _MinHash
    from datasketch import MinHashLSH as _MinHashLSH

    _HAS_DATASKETCH = True
except ImportError:
    _MinHash = None  # type: ignore[assignment,misc]
    _MinHashLSH = None  # type: ignore[assignment,misc]
    _HAS_DATASKETCH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SHINGLE_K: int = 3
_DEFAULT_NUM_PERM: int = 128
_DEFAULT_THRESHOLD: float = 0.5
_SMALL_SET_CUTOFF: int = 500

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DedupResult:
    """A group of near-duplicate items with their canonical form."""

    canonical: str
    variants: tuple[str, ...]
    similarity: float


@dataclass(frozen=True)
class DocumentFingerprint:
    """Fingerprint of an indexed document."""

    doc_id: str
    shingle_count: int


@dataclass(frozen=True)
class DuplicateMatch:
    """A match found in the dedup index."""

    doc_id: str
    similarity: float


# ---------------------------------------------------------------------------
# Shingling
# ---------------------------------------------------------------------------


def shingle_text(text: str, k: int = _DEFAULT_SHINGLE_K) -> frozenset[str]:
    """Create k-word shingles from text (lowercased)."""
    words = text.lower().split()
    if len(words) < k:
        return frozenset()
    return frozenset(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))


# ---------------------------------------------------------------------------
# MinHash
# ---------------------------------------------------------------------------


def compute_minhash(
    shingles: frozenset[str], num_perm: int = _DEFAULT_NUM_PERM
) -> object:
    """Compute MinHash signature for a shingle set.

    Returns a datasketch MinHash when available, otherwise the raw
    frozenset for direct Jaccard computation.
    """
    if not shingles:
        if _HAS_DATASKETCH:
            return _MinHash(num_perm=num_perm)
        return frozenset()

    if _HAS_DATASKETCH:
        m = _MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))
        return m

    return shingles  # fallback: raw shingles for Jaccard


def estimate_similarity(sig1: object, sig2: object) -> float:
    """Estimate Jaccard similarity between two signatures."""
    if _HAS_DATASKETCH and isinstance(sig1, _MinHash) and isinstance(sig2, _MinHash):
        return float(sig1.jaccard(sig2))

    # Fallback: exact Jaccard on frozensets
    s1 = sig1 if isinstance(sig1, frozenset) else frozenset()
    s2 = sig2 if isinstance(sig2, frozenset) else frozenset()
    if not s1 and not s2:
        return 0.0
    union = s1 | s2
    if not union:
        return 0.0
    return len(s1 & s2) / len(union)


# ---------------------------------------------------------------------------
# DedupIndex
# ---------------------------------------------------------------------------


@dataclass
class DedupIndex:
    """Near-duplicate index using MinHash/LSH or pairwise fallback."""

    threshold: float = _DEFAULT_THRESHOLD
    _num_perm: int = _DEFAULT_NUM_PERM
    _signatures: dict[str, object] = field(default_factory=dict, repr=False)
    _shingles: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)
    _lsh: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if _HAS_DATASKETCH:
            self._lsh = _MinHashLSH(threshold=self.threshold, num_perm=self._num_perm)

    def add(self, doc_id: str, text: str) -> DocumentFingerprint:
        """Index a document and return its fingerprint."""
        shingles = shingle_text(text)
        sig = compute_minhash(shingles, self._num_perm)

        self._signatures[doc_id] = sig
        self._shingles[doc_id] = shingles

        if self._lsh is not None and _HAS_DATASKETCH and isinstance(sig, _MinHash):
            try:
                self._lsh.insert(doc_id, sig)  # type: ignore[union-attr]
            except ValueError:
                pass  # duplicate key

        return DocumentFingerprint(doc_id=doc_id, shingle_count=len(shingles))

    def find_duplicates(self, doc_id: str) -> list[DuplicateMatch]:
        """Find near-duplicates of a document already in the index."""
        if doc_id not in self._signatures:
            return []

        sig = self._signatures[doc_id]
        matches: list[DuplicateMatch] = []

        if self._lsh is not None and _HAS_DATASKETCH and isinstance(sig, _MinHash):
            candidates = self._lsh.query(sig)  # type: ignore[union-attr]
            for cand_id in candidates:
                if cand_id == doc_id:
                    continue
                sim = estimate_similarity(sig, self._signatures[cand_id])
                if sim >= self.threshold:
                    matches.append(DuplicateMatch(doc_id=cand_id, similarity=sim))
        else:
            # Fallback: pairwise comparison
            for other_id, other_sig in self._signatures.items():
                if other_id == doc_id:
                    continue
                sim = estimate_similarity(sig, other_sig)
                if sim >= self.threshold:
                    matches.append(DuplicateMatch(doc_id=other_id, similarity=sim))

        return sorted(matches, key=lambda m: -m.similarity)

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a near-duplicate of anything in the index."""
        shingles = shingle_text(text)
        sig = compute_minhash(shingles, self._num_perm)

        if self._lsh is not None and _HAS_DATASKETCH and isinstance(sig, _MinHash):
            candidates = self._lsh.query(sig)  # type: ignore[union-attr]
            return len(candidates) > 0

        # Fallback: pairwise
        for other_sig in self._signatures.values():
            sim = estimate_similarity(sig, other_sig)
            if sim >= self.threshold:
                return True
        return False

    def deduplicate(self) -> list[DedupResult]:
        """Group all indexed documents into dedup clusters."""
        visited: set[str] = set()
        groups: list[DedupResult] = []

        for doc_id in self._signatures:
            if doc_id in visited:
                continue
            visited.add(doc_id)

            dupes = self.find_duplicates(doc_id)
            variants = []
            best_sim = 0.0
            for d in dupes:
                if d.doc_id not in visited:
                    visited.add(d.doc_id)
                    variants.append(d.doc_id)
                    best_sim = max(best_sim, d.similarity)

            groups.append(DedupResult(
                canonical=doc_id,
                variants=tuple(variants),
                similarity=best_sim,
            ))

        return groups


# ---------------------------------------------------------------------------
# Entity dedup bridge
# ---------------------------------------------------------------------------


def build_entity_mappings_minhash(
    entities: list[str],
    existing_mappings: dict[str, str],
    threshold: float = _DEFAULT_THRESHOLD,
) -> dict[str, str]:
    """Build variant→canonical entity mappings using MinHash for large sets.

    For sets smaller than _SMALL_SET_CUTOFF, delegates to the existing
    difflib-based `build_fuzzy_mappings()` to preserve exact behavior.
    """
    if not entities:
        return {}

    if len(entities) < _SMALL_SET_CUTOFF:
        from src.data.entity_filter import build_fuzzy_mappings

        return build_fuzzy_mappings(entities, existing_mappings, threshold)

    # Large set: MinHash/LSH approach
    candidates = [e for e in entities if e not in existing_mappings]
    if not candidates:
        return {}

    idx = DedupIndex(threshold=threshold)
    for entity in candidates:
        idx.add(entity, entity)  # use entity text as its own document

    groups = idx.deduplicate()
    new_mappings: dict[str, str] = {}

    existing_values = set(existing_mappings.values())

    for group in groups:
        if not group.variants:
            continue

        all_members = [group.canonical, *group.variants]

        # Pick canonical: prefer existing mapping value > longer > alphabetical
        canonical = None
        for member in all_members:
            if member in existing_values:
                canonical = member
                break
        if canonical is None:
            canonical = sorted(all_members, key=lambda n: (-len(n), n))[0]

        for member in all_members:
            if member != canonical:
                new_mappings[member] = canonical

    return new_mappings
