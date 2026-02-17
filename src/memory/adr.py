from __future__ import annotations

from dataclasses import dataclass

from src.memory.vault import VaultClient

_DECISIONS_PREFIX = "decisions"
_TRACES_PREFIX = "hypothesis-traces"
_VALID_STATUSES: frozenset[str] = frozenset({"proposed", "accepted", "deprecated", "superseded"})


def _yaml_str(value: str) -> str:
    """Quote a scalar for YAML — prevents colon/hash/newline injection in frontmatter."""
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


@dataclass(frozen=True)
class ADR:
    """Architecture Decision Record.

    Frozen so that ADR identity is stable across the session — mutations would
    invalidate any vault entry already written under this id.
    """

    id: str
    title: str
    status: str
    context: str
    decision: str
    consequences: str
    files: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid ADR status {self.status!r}. Must be one of: {sorted(_VALID_STATUSES)}"
            )


@dataclass(frozen=True)
class HypothesisTrace:
    """Point-in-time snapshot of a hypothesis and its linked analytical gaps.

    Immutable so that the audit trail of reasoning remains intact even as
    hypotheses evolve through dataclasses.replace() spawning.
    """

    hypothesis_id: str
    hypothesis_text: str
    confidence: float
    evidence: str
    gap_ids: tuple[str, ...]
    parent_id: str | None
    timestamp: str


def render_adr(adr: ADR) -> str:
    """Produce Obsidian-compatible markdown from a structured ADR.

    Frontmatter first so Obsidian indexes metadata without parsing the body.
    """
    if adr.tags:
        tags_block = "\n" + "".join(f"  - {_yaml_str(t)}\n" for t in adr.tags)
    else:
        tags_block = " []\n"
    files_block = "\n".join(adr.files) if adr.files else "(none)"
    return (
        f"---\n"
        f"id: {_yaml_str(adr.id)}\n"
        f"title: {_yaml_str(adr.title)}\n"
        f"status: {_yaml_str(adr.status)}\n"
        f"tags:{tags_block}"
        f"---\n"
        f"\n"
        f"## Decision\n"
        f"\n"
        f"{adr.decision}\n"
        f"\n"
        f"## Context\n"
        f"\n"
        f"{adr.context}\n"
        f"\n"
        f"## Consequences\n"
        f"\n"
        f"{adr.consequences}\n"
        f"\n"
        f"## Files\n"
        f"\n"
        f"{files_block}\n"
    )


def render_hypothesis_trace(trace: HypothesisTrace) -> str:
    """Produce Obsidian-compatible markdown from a HypothesisTrace snapshot.

    Parent linkage and timestamp in frontmatter support Obsidian Dataview
    queries over the hypothesis evolution graph.
    """
    gaps_block = "\n".join(trace.gap_ids) if trace.gap_ids else "(none)"
    parent_value = trace.parent_id if trace.parent_id is not None else "null"
    return (
        f"---\n"
        f"hypothesis_id: {_yaml_str(trace.hypothesis_id)}\n"
        f"confidence: {trace.confidence}\n"
        f"parent_id: {_yaml_str(parent_value)}\n"
        f"timestamp: {_yaml_str(trace.timestamp)}\n"
        f"---\n"
        f"\n"
        f"## Hypothesis\n"
        f"\n"
        f"{trace.hypothesis_text}\n"
        f"\n"
        f"## Evidence\n"
        f"\n"
        f"{trace.evidence}\n"
        f"\n"
        f"## Linked Gaps\n"
        f"\n"
        f"{gaps_block}\n"
    )


def persist_adr(adr: ADR, vault: VaultClient) -> None:
    """Write a rendered ADR into the vault's decisions subtree.

    Separating rendering from I/O keeps render_adr() pure and testable without
    a real vault.
    """
    path = f"{_DECISIONS_PREFIX}/{adr.id}.md"
    vault.write_note(path, render_adr(adr))


def persist_hypothesis_trace(trace: HypothesisTrace, vault: VaultClient) -> None:
    """Write a rendered HypothesisTrace into the vault's hypothesis-traces subtree."""
    path = f"{_TRACES_PREFIX}/{trace.hypothesis_id}.md"
    vault.write_note(path, render_hypothesis_trace(trace))
