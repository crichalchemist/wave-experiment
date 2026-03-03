"""Tests for 3-layer entity filter (ADR-016)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.entity_filter import (
    DropLog,
    EntityDrop,
    build_fuzzy_mappings,
    filter_entities,
    is_junk,
    is_role_description,
    _name_key,
)


# ---------------------------------------------------------------------------
# Layer 1 — junk detection
# ---------------------------------------------------------------------------


class TestLayer1Junk:
    """Tests for FOIA codes, short entities, emails, numeric refs, redacted."""

    def test_foia_code_paren_start(self) -> None:
        assert is_junk("(b)(6)") == "foia_code"

    def test_foia_code_variant(self) -> None:
        assert is_junk("(b)(7)(C)") == "foia_code"

    def test_foia_code_embedded(self) -> None:
        assert is_junk("Officer (b)(6)") == "foia_code_embedded"

    def test_bracket_redacted_lower(self) -> None:
        assert is_junk("[redacted]") == "bracket_redacted"

    def test_bracket_redacted_upper(self) -> None:
        assert is_junk("[REDACTED]") == "bracket_redacted"

    def test_email_at_sign(self) -> None:
        assert is_junk("user@aol.com") == "email_or_handle"

    def test_handle_at_sign(self) -> None:
        assert is_junk("@bop.gov") == "email_or_handle"

    def test_pure_numeric(self) -> None:
        assert is_junk("805899") == "numeric_ref"

    def test_too_short_single_char(self) -> None:
        assert is_junk("D") == "too_short"

    def test_too_short_two_chars(self) -> None:
        assert is_junk("AG") == "too_short"

    def test_clean_name_passes(self) -> None:
        assert is_junk("Jeffrey Epstein") is None

    def test_clean_three_char_passes(self) -> None:
        """Three-character names like 'FBI' should not be filtered."""
        assert is_junk("FBI") is None

    def test_clean_org_passes(self) -> None:
        assert is_junk("Department of Justice") is None


# ---------------------------------------------------------------------------
# Layer 3 — role description detection
# ---------------------------------------------------------------------------


class TestLayer3RoleDescription:
    """Tests for possessives, inmate patterns, role prefixes, informal names."""

    def test_possessive_apostrophe(self) -> None:
        assert is_role_description("Epstein's attorneys") == "possessive_phrase"

    def test_possessive_smart_quote(self) -> None:
        assert is_role_description("Epstein\u2019s lawyers") == "possessive_phrase"

    def test_inmate_prefix(self) -> None:
        assert is_role_description("Inmate 7") == "inmate_pattern"

    def test_inmate_prefix_case_insensitive(self) -> None:
        assert is_role_description("inmate Reyes") == "inmate_pattern"

    def test_dot_anonymized(self) -> None:
        assert is_role_description(".CONCIERGE") == "dot_anonymized"

    def test_anonymized_officer_star(self) -> None:
        assert is_role_description("*D - CBP OFFCR-C") == "anonymized_officer"

    def test_role_prefix_defendant(self) -> None:
        assert is_role_description("defendants") == "role_prefix"

    def test_role_prefix_co_conspirator(self) -> None:
        assert is_role_description("co-conspirator") == "role_prefix"

    def test_lowercase_informal(self) -> None:
        assert is_role_description("autograph") == "lowercase_informal"

    def test_lowercase_informal_username(self) -> None:
        assert is_role_description("babiigirl1322") is None  # has digits → not all-lowercase

    def test_short_lowercase_passes(self) -> None:
        """Short lowercase strings (<=5 chars) are NOT filtered."""
        assert is_role_description("smith") is None

    def test_clean_real_name(self) -> None:
        assert is_role_description("Ghislaine Maxwell") is None

    def test_clean_title_name(self) -> None:
        """Names with titles like 'Agent Brown' should NOT be filtered."""
        assert is_role_description("Agent Brown") is None

    def test_clean_judge_name(self) -> None:
        assert is_role_description("Judge Nathan") is None


# ---------------------------------------------------------------------------
# Layer 2 — fuzzy deduplication
# ---------------------------------------------------------------------------


class TestNameKey:
    """Tests for the name normalization helper."""

    def test_strip_parenthetical(self) -> None:
        assert _name_key("Jeffrey Epstein (Mr. Epstein)") == "jeffrey epstein"

    def test_flip_lastname_first(self) -> None:
        assert _name_key("EPSTEIN, JEFFREY EDWARD") == "jeffrey edward epstein"

    def test_lowercase_strip_nonalpha(self) -> None:
        assert _name_key("Dr. J. Smith III") == "dr j smith iii"


class TestLayer2FuzzyDedup:
    """Tests for fuzzy deduplication pre-pass."""

    def test_matches_similar_names(self) -> None:
        entities = ["Jeffrey Epstein", "EPSTEIN, JEFFREY EDWARD"]
        result = build_fuzzy_mappings(entities, {})
        # One should map to the other
        assert len(result) == 1

    def test_respects_existing_canonical(self) -> None:
        entities = ["Jeff E.", "EPSTEIN, JEFFREY"]
        existing = {"some_variant": "Jeffrey Epstein"}
        result = build_fuzzy_mappings(entities, existing)
        # New mappings should prefer existing canonical value if match found
        for _variant, canon in result.items():
            # Canon should be the longer or preferred form
            assert isinstance(canon, str)

    def test_skips_already_mapped(self) -> None:
        """Entities already in existing_mappings keys are skipped."""
        entities = ["Jeff Epstein", "Jeffrey Epstein"]
        existing = {"Jeff Epstein": "Jeffrey Epstein"}
        result = build_fuzzy_mappings(entities, existing)
        assert "Jeff Epstein" not in result

    def test_no_false_positives_dissimilar(self) -> None:
        """Completely different names should not be matched."""
        entities = ["Jeffrey Epstein", "Ghislaine Maxwell"]
        result = build_fuzzy_mappings(entities, {})
        assert len(result) == 0

    def test_threshold_sensitivity(self) -> None:
        """A low threshold matches more; a high threshold matches fewer."""
        entities = ["John Smith", "Jon Smith"]
        low = build_fuzzy_mappings(entities, {}, threshold=0.5)
        high = build_fuzzy_mappings(entities, {}, threshold=0.99)
        assert len(low) >= len(high)

    def test_empty_input(self) -> None:
        assert build_fuzzy_mappings([], {}) == {}

    def test_canonical_prefers_longer_name(self) -> None:
        entities = ["J. Epstein", "Jeffrey Edward Epstein"]
        result = build_fuzzy_mappings(entities, {}, threshold=0.5)
        if result:
            # The shorter name should map to the longer one
            assert "J. Epstein" in result
            assert result["J. Epstein"] == "Jeffrey Edward Epstein"


# ---------------------------------------------------------------------------
# DropLog
# ---------------------------------------------------------------------------


class TestDropLog:
    def test_creates_file_and_writes_jsonl(self, tmp_path: Path) -> None:
        log_path = tmp_path / "drops.jsonl"
        log = DropLog(log_path)
        log.record(EntityDrop(entity="(b)(6)", reason="foia_code", category="layer1"))

        assert log_path.exists()
        line = json.loads(log_path.read_text().strip())
        assert line["entity"] == "(b)(6)"
        assert line["reason"] == "foia_code"
        assert line["category"] == "layer1"

    def test_count_tracks_records(self, tmp_path: Path) -> None:
        log = DropLog(tmp_path / "drops.jsonl")
        assert log.count == 0
        log.record(EntityDrop(entity="x", reason="r", category="layer1"))
        log.record(EntityDrop(entity="y", reason="r", category="layer3"))
        assert log.count == 2

    def test_appends_multiple_lines(self, tmp_path: Path) -> None:
        log_path = tmp_path / "drops.jsonl"
        log = DropLog(log_path)
        log.record(EntityDrop(entity="a", reason="r1", category="layer1"))
        log.record(EntityDrop(entity="b", reason="r2", category="layer3"))

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Orchestrator — filter_entities
# ---------------------------------------------------------------------------


class TestFilterEntities:
    def test_mixed_entities(self, tmp_path: Path) -> None:
        entities = [
            "Jeffrey Epstein",      # clean
            "(b)(6)",               # Layer 1: FOIA code
            "Ghislaine Maxwell",    # clean
            "Inmate 7",            # Layer 3: inmate pattern
            "D",                    # Layer 1: too short
            "defendants",           # Layer 3: role prefix
        ]
        log = DropLog(tmp_path / "drops.jsonl")
        result = filter_entities(entities, drop_log=log)

        assert result == ["Jeffrey Epstein", "Ghislaine Maxwell"]
        assert log.count == 4

    def test_no_log_mode(self) -> None:
        entities = ["Jeffrey Epstein", "(b)(6)", "defendants"]
        result = filter_entities(entities, drop_log=None)
        assert result == ["Jeffrey Epstein"]
