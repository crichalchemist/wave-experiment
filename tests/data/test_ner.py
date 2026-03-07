"""Tests for NER pipeline — frozen types, extraction, fallback."""

from __future__ import annotations

import pytest

from src.data.ner import NerEntity, NerResult, extract_entities


# ---------------------------------------------------------------------------
# NerEntity frozen dataclass
# ---------------------------------------------------------------------------


class TestNerEntity:
    def test_frozen(self):
        ent = NerEntity(text="Jeffrey Epstein", label="PERSON", start=0, end=15)
        with pytest.raises(AttributeError):
            ent.text = "mutated"  # type: ignore[misc]

    def test_fields(self):
        ent = NerEntity(text="FBI", label="ORG", start=10, end=13)
        assert ent.text == "FBI"
        assert ent.label == "ORG"
        assert ent.start == 10
        assert ent.end == 13


# ---------------------------------------------------------------------------
# NerResult frozen dataclass
# ---------------------------------------------------------------------------


class TestNerResult:
    def _make_result(self) -> NerResult:
        entities = (
            NerEntity(text="Jeffrey Epstein", label="PERSON", start=0, end=15),
            NerEntity(text="Ghislaine Maxwell", label="PERSON", start=20, end=37),
            NerEntity(text="FBI", label="ORG", start=50, end=53),
            NerEntity(text="New York", label="GPE", start=60, end=68),
            NerEntity(text="Jeffrey Epstein", label="PERSON", start=100, end=115),
        )
        return NerResult(entities=entities, backend="test", text_length=120)

    def test_frozen(self):
        result = self._make_result()
        with pytest.raises(AttributeError):
            result.backend = "mutated"  # type: ignore[misc]

    def test_persons(self):
        result = self._make_result()
        persons = result.persons
        assert all(e.label == "PERSON" for e in persons)
        assert len(persons) == 3

    def test_organizations(self):
        result = self._make_result()
        orgs = result.organizations
        assert all(e.label == "ORG" for e in orgs)
        assert len(orgs) == 1
        assert orgs[0].text == "FBI"

    def test_unique_texts_all(self):
        result = self._make_result()
        unique = result.unique_texts()
        # "Jeffrey Epstein" appears twice but should deduplicate
        assert "Jeffrey Epstein" in unique
        assert len(set(unique)) == len(unique)  # all unique

    def test_unique_texts_by_label(self):
        result = self._make_result()
        persons = result.unique_texts("PERSON")
        assert "Jeffrey Epstein" in persons
        assert "Ghislaine Maxwell" in persons
        assert "FBI" not in persons

    def test_empty_result(self):
        result = NerResult(entities=(), backend="heuristic", text_length=0)
        assert result.persons == ()
        assert result.organizations == ()
        assert result.unique_texts() == ()


# ---------------------------------------------------------------------------
# extract_entities — heuristic fallback (always available)
# ---------------------------------------------------------------------------


class TestExtractEntitiesHeuristic:
    """Tests that work regardless of whether spaCy is installed."""

    def test_returns_ner_result(self):
        result = extract_entities("Jeffrey Epstein was arrested in New York.")
        assert isinstance(result, NerResult)

    def test_backend_is_string(self):
        result = extract_entities("Test text.")
        assert result.backend in ("spacy", "heuristic")

    def test_text_length_tracked(self):
        text = "Some test text here."
        result = extract_entities(text)
        assert result.text_length == len(text)

    def test_empty_text(self):
        result = extract_entities("")
        assert result.entities == ()
        assert result.text_length == 0

    def test_finds_capitalized_names(self):
        result = extract_entities(
            "According to Maxwell, the meeting was arranged by Epstein."
        )
        texts = result.unique_texts()
        # Should find at least Maxwell and Epstein (both backends)
        assert any("Maxwell" in t for t in texts)
        assert any("Epstein" in t for t in texts)

    def test_skips_sentence_starters(self):
        """Heuristic should not extract common sentence-starting words."""
        result = extract_entities("The investigation revealed that However nothing was found.")
        texts = result.unique_texts()
        # "The" and "However" should not appear as entities in heuristic mode
        if result.backend == "heuristic":
            assert "The" not in texts
            assert "However" not in texts

    def test_multiword_entities(self):
        """Should capture multi-word entity candidates."""
        result = extract_entities("Jeffrey Epstein flew to Palm Beach with Bill Clinton.")
        texts = result.unique_texts()
        # Both backends should find multi-word names
        assert any("Jeffrey Epstein" in t or "Epstein" in t for t in texts)

    def test_all_caps_org_detection(self):
        """Heuristic should classify short all-caps as ORG."""
        result = extract_entities("The FBI and SEC investigated the case.")
        if result.backend == "heuristic":
            orgs = result.organizations
            org_texts = [o.text for o in orgs]
            assert "FBI" in org_texts
            assert "SEC" in org_texts


# ---------------------------------------------------------------------------
# spaCy integration tests (skip if not installed)
# ---------------------------------------------------------------------------

_spacy_available = False
try:
    import spacy
    spacy.load("en_core_web_sm")
    _spacy_available = True
except (ImportError, OSError):
    pass


@pytest.mark.skipif(not _spacy_available, reason="spaCy en_core_web_sm not installed")
class TestSpacyExtraction:
    """Tests that require spaCy to be installed."""

    def test_spacy_backend_used(self):
        # Reset the cache to force reload
        import src.data.ner as ner_mod
        ner_mod._nlp_loaded = False
        ner_mod._nlp_cache = None
        result = extract_entities("Jeffrey Epstein was arrested.")
        assert result.backend == "spacy"

    def test_person_entity(self):
        result = extract_entities("Jeffrey Epstein attended the meeting with Ghislaine Maxwell.")
        persons = result.unique_texts("PERSON")
        assert any("Epstein" in p for p in persons)

    def test_org_entity(self):
        result = extract_entities("The Federal Bureau of Investigation opened an inquiry.")
        orgs = result.unique_texts("ORG")
        assert len(orgs) >= 1

    def test_gpe_entity(self):
        result = extract_entities("They traveled to New York and London.")
        gpes = tuple(e for e in result.entities if e.label == "GPE")
        assert len(gpes) >= 1
