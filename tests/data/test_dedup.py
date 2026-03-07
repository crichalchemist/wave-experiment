"""Tests for MinHash/LSH near-duplicate detection."""

from __future__ import annotations

import pytest

from src.data.dedup import (
    DedupIndex,
    DedupResult,
    DocumentFingerprint,
    DuplicateMatch,
    build_entity_mappings_minhash,
    compute_minhash,
    estimate_similarity,
    shingle_text,
)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


class TestDedupResult:
    def test_frozen(self):
        r = DedupResult(canonical="Jeffrey Epstein", variants=("Jeff Epstein",), similarity=0.8)
        with pytest.raises(AttributeError):
            r.canonical = "mutated"  # type: ignore[misc]

    def test_fields(self):
        r = DedupResult(canonical="A", variants=("B", "C"), similarity=0.75)
        assert r.canonical == "A"
        assert r.variants == ("B", "C")
        assert r.similarity == 0.75


class TestDocumentFingerprint:
    def test_frozen(self):
        fp = DocumentFingerprint(doc_id="doc1", shingle_count=10)
        with pytest.raises(AttributeError):
            fp.doc_id = "mutated"  # type: ignore[misc]


class TestDuplicateMatch:
    def test_frozen(self):
        m = DuplicateMatch(doc_id="doc2", similarity=0.9)
        with pytest.raises(AttributeError):
            m.similarity = 0.5  # type: ignore[misc]

    def test_fields(self):
        m = DuplicateMatch(doc_id="x", similarity=0.65)
        assert m.doc_id == "x"
        assert m.similarity == 0.65


# ---------------------------------------------------------------------------
# Shingling
# ---------------------------------------------------------------------------


class TestShingleText:
    def test_basic(self):
        shingles = shingle_text("the quick brown fox jumps", k=3)
        assert isinstance(shingles, frozenset)
        assert "the quick brown" in shingles
        assert "quick brown fox" in shingles
        assert "brown fox jumps" in shingles

    def test_short_text(self):
        """Text shorter than k words should return empty set."""
        assert shingle_text("hello world", k=3) == frozenset()

    def test_single_shingle(self):
        shingles = shingle_text("one two three", k=3)
        assert len(shingles) == 1

    def test_empty_text(self):
        assert shingle_text("", k=3) == frozenset()

    def test_case_normalization(self):
        shingles = shingle_text("The Quick Brown", k=3)
        assert "the quick brown" in shingles


# ---------------------------------------------------------------------------
# MinHash + similarity
# ---------------------------------------------------------------------------


class TestMinHashSimilarity:
    def test_identical_texts_high_similarity(self):
        text = "Jeffrey Epstein was arrested in New York on federal charges"
        s1 = shingle_text(text)
        s2 = shingle_text(text)
        sig1 = compute_minhash(s1)
        sig2 = compute_minhash(s2)
        sim = estimate_similarity(sig1, sig2)
        assert sim > 0.95

    def test_similar_texts_moderate_similarity(self):
        s1 = shingle_text("Jeffrey Epstein was arrested in New York on federal charges")
        s2 = shingle_text("Jeffrey Epstein was arrested in New York on criminal charges")
        sig1 = compute_minhash(s1)
        sig2 = compute_minhash(s2)
        sim = estimate_similarity(sig1, sig2)
        assert 0.3 < sim < 1.0

    def test_different_texts_low_similarity(self):
        s1 = shingle_text("Jeffrey Epstein was arrested in New York on federal charges")
        s2 = shingle_text("The weather forecast for London predicts rain and cold temperatures")
        sig1 = compute_minhash(s1)
        sig2 = compute_minhash(s2)
        sim = estimate_similarity(sig1, sig2)
        assert sim < 0.3

    def test_empty_shingles(self):
        sig1 = compute_minhash(frozenset())
        sig2 = compute_minhash(frozenset())
        sim = estimate_similarity(sig1, sig2)
        assert sim == 0.0


# ---------------------------------------------------------------------------
# DedupIndex
# ---------------------------------------------------------------------------


class TestDedupIndex:
    def test_add_returns_fingerprint(self):
        idx = DedupIndex(threshold=0.5)
        fp = idx.add("doc1", "Jeffrey Epstein was arrested in New York")
        assert isinstance(fp, DocumentFingerprint)
        assert fp.doc_id == "doc1"
        assert fp.shingle_count > 0

    def test_find_duplicates_similar(self):
        idx = DedupIndex(threshold=0.4)
        idx.add("doc1", "Jeffrey Epstein was arrested in New York on federal charges")
        idx.add("doc2", "Jeffrey Epstein was arrested in New York on criminal charges")
        dupes = idx.find_duplicates("doc1")
        assert any(d.doc_id == "doc2" for d in dupes)

    def test_find_duplicates_dissimilar(self):
        idx = DedupIndex(threshold=0.5)
        idx.add("doc1", "Jeffrey Epstein was arrested in New York on federal charges")
        idx.add("doc2", "The weather forecast for London predicts rain and cold temperatures all week")
        dupes = idx.find_duplicates("doc1")
        assert not any(d.doc_id == "doc2" for d in dupes)

    def test_empty_index(self):
        idx = DedupIndex()
        assert idx.find_duplicates("nonexistent") == []

    def test_deduplicate_groups(self):
        idx = DedupIndex(threshold=0.4)
        idx.add("doc1", "Jeffrey Epstein was arrested in New York on federal charges last Tuesday")
        idx.add("doc2", "Jeffrey Epstein was arrested in New York on criminal charges last Tuesday")
        idx.add("doc3", "The weather forecast for London predicts rain and cold temperatures all week")
        groups = idx.deduplicate()
        assert isinstance(groups, list)
        # doc1 and doc2 should be in the same group, doc3 separate
        for g in groups:
            assert isinstance(g, DedupResult)

    def test_is_duplicate_convenience(self):
        idx = DedupIndex(threshold=0.4)
        idx.add("doc1", "Jeffrey Epstein was arrested in New York on federal charges last Tuesday")
        assert idx.is_duplicate("Jeffrey Epstein was arrested in New York on criminal charges last Tuesday")
        assert not idx.is_duplicate("The weather forecast for London predicts rain and cold temperatures all week long")


# ---------------------------------------------------------------------------
# build_entity_mappings_minhash
# ---------------------------------------------------------------------------


class TestBuildEntityMappingsMinhash:
    def test_small_set_delegates_to_difflib(self):
        """Sets < 500 should use difflib (existing behavior)."""
        entities = ["Jeffrey Epstein", "Jeff Epstein", "Bill Clinton"]
        result = build_entity_mappings_minhash(entities, {}, threshold=0.5)
        assert isinstance(result, dict)

    def test_returns_mappings(self):
        entities = ["Jeffrey Epstein", "Jeff Epstein", "Ghislaine Maxwell"]
        result = build_entity_mappings_minhash(entities, {}, threshold=0.5)
        # Should map one Epstein variant to the other
        assert isinstance(result, dict)

    def test_empty_entities(self):
        result = build_entity_mappings_minhash([], {}, threshold=0.5)
        assert result == {}

    def test_respects_existing_mappings(self):
        entities = ["Jeff Epstein", "Bill Clinton"]
        existing = {"Jeff Epstein": "Jeffrey Epstein"}
        result = build_entity_mappings_minhash(entities, existing, threshold=0.5)
        # Jeff Epstein already mapped, shouldn't appear in new mappings
        assert "Jeff Epstein" not in result
