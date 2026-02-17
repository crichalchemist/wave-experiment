from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from src.memory.adr import (
    ADR,
    HypothesisTrace,
    persist_adr,
    persist_hypothesis_trace,
    render_adr,
    render_hypothesis_trace,
)
from src.memory.vault import VaultClient


# --- Fixtures ---

@pytest.fixture
def sample_adr() -> ADR:
    return ADR(
        id="ADR-009",
        title="Some Decision",
        status="accepted",
        context="We needed a way to persist structured decisions.",
        decision="Use Obsidian-compatible markdown with YAML frontmatter.",
        consequences="All ADRs are now readable in Obsidian and version-controlled.",
        files=("src/memory/adr.py", "tests/memory/test_adr.py"),
        tags=("architecture", "memory"),
    )


@pytest.fixture
def sample_trace() -> HypothesisTrace:
    return HypothesisTrace(
        hypothesis_id="H-001",
        hypothesis_text="Entity A coordinated with Entity B between 2010 and 2014.",
        confidence=0.72,
        evidence="Three documents reference joint operations during this period.",
        gap_ids=("GAP-001", "GAP-003"),
        parent_id="H-000",
        timestamp="2026-02-17T12:00:00",
    )


@pytest.fixture
def trace_no_gaps() -> HypothesisTrace:
    return HypothesisTrace(
        hypothesis_id="H-002",
        hypothesis_text="Entity C acted independently.",
        confidence=0.50,
        evidence="No corroborating links found.",
        gap_ids=(),
        parent_id=None,
        timestamp="2026-02-17T13:00:00",
    )


# --- ADR rendering tests ---

def test_render_adr_contains_frontmatter(sample_adr: ADR) -> None:
    rendered = render_adr(sample_adr)
    assert rendered.startswith("---\n")
    assert 'id: "ADR-009"' in rendered
    assert 'title: "Some Decision"' in rendered
    # YAML block closes before body content
    parts = rendered.split("---")
    assert len(parts) >= 3  # opening ---, frontmatter, closing ---


def test_render_adr_sections_present(sample_adr: ADR) -> None:
    rendered = render_adr(sample_adr)
    assert "## Decision" in rendered
    assert "## Context" in rendered
    assert "## Consequences" in rendered
    assert "## Files" in rendered


def test_render_adr_files_listed(sample_adr: ADR) -> None:
    rendered = render_adr(sample_adr)
    for file_path in sample_adr.files:
        assert file_path in rendered


def test_render_adr_tags_in_frontmatter(sample_adr: ADR) -> None:
    rendered = render_adr(sample_adr)
    # Tags must appear within the YAML frontmatter block (between first pair of ---)
    frontmatter_end = rendered.index("---", 3)
    frontmatter = rendered[:frontmatter_end]
    assert "architecture" in frontmatter
    assert "memory" in frontmatter


def test_render_adr_tags_block_style(sample_adr: ADR) -> None:
    rendered = render_adr(sample_adr)
    # Block-style tags: each tag on its own line with "  - " prefix
    assert '  - "architecture"' in rendered
    assert '  - "memory"' in rendered


def test_render_adr_empty_tags_inline(sample_adr: ADR) -> None:
    adr_no_tags = ADR(
        id="ADR-010",
        title="No Tags",
        status="proposed",
        context="ctx",
        decision="dec",
        consequences="cons",
        files=(),
        tags=(),
    )
    rendered = render_adr(adr_no_tags)
    assert "tags: []" in rendered


def test_render_adr_yaml_safe_title() -> None:
    """A colon-space in the title must not break YAML frontmatter."""
    adr = ADR(
        id="ADR-011",
        title="Use httpx: async client",
        status="accepted",
        context="ctx",
        decision="dec",
        consequences="cons",
        files=(),
        tags=(),
    )
    rendered = render_adr(adr)
    assert 'title: "Use httpx: async client"' in rendered


# --- HypothesisTrace rendering tests ---

def test_render_hypothesis_trace_contains_frontmatter(sample_trace: HypothesisTrace) -> None:
    rendered = render_hypothesis_trace(sample_trace)
    assert rendered.startswith("---\n")
    assert 'hypothesis_id: "H-001"' in rendered
    assert "confidence: 0.72" in rendered
    assert 'timestamp: "2026-02-17T12:00:00"' in rendered


def test_render_hypothesis_trace_sections(sample_trace: HypothesisTrace) -> None:
    rendered = render_hypothesis_trace(sample_trace)
    assert "## Hypothesis" in rendered
    assert "## Evidence" in rendered
    assert "## Linked Gaps" in rendered


def test_render_hypothesis_trace_no_gaps(trace_no_gaps: HypothesisTrace) -> None:
    rendered = render_hypothesis_trace(trace_no_gaps)
    assert "(none)" in rendered
    assert "## Linked Gaps" in rendered
    assert 'parent_id: "null"' in rendered


# --- Vault persistence tests ---

def test_persist_adr_writes_to_correct_path(sample_adr: ADR) -> None:
    mock_vault = MagicMock(spec=VaultClient)
    persist_adr(sample_adr, mock_vault)
    mock_vault.write_note.assert_called_once()
    call_path, call_content = mock_vault.write_note.call_args[0]
    assert call_path == "decisions/ADR-009.md"
    assert "ADR-009" in call_content


def test_persist_hypothesis_trace_writes_to_correct_path(sample_trace: HypothesisTrace) -> None:
    mock_vault = MagicMock(spec=VaultClient)
    persist_hypothesis_trace(sample_trace, mock_vault)
    mock_vault.write_note.assert_called_once()
    call_path, call_content = mock_vault.write_note.call_args[0]
    assert call_path == "hypothesis-traces/H-001.md"
    assert "H-001" in call_content


# --- Immutability tests ---

def test_adr_is_frozen(sample_adr: ADR) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample_adr.title = "Mutated Title"  # type: ignore[misc]


def test_hypothesis_trace_is_frozen(sample_trace: HypothesisTrace) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample_trace.confidence = 0.99  # type: ignore[misc]


# --- Status validation tests ---

def test_adr_invalid_status_raises() -> None:
    with pytest.raises(ValueError, match="Invalid ADR status"):
        ADR(
            id="ADR-001",
            title="Test",
            status="actived",  # typo — should be "accepted"
            context="ctx",
            decision="dec",
            consequences="cons",
            files=(),
            tags=(),
        )
