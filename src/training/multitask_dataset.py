"""Multi-task dataset loading and tokenization for DetectiveGPT.

Provides JSONL annotation loading, character-level tokenization (BOS convention
from microgpt.py), and label-index mappings for gap and assumption types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.core.types import AssumptionType, GapType

# Label-index mappings (dense integer labels for CrossEntropyLoss)
GAP_TYPE_TO_INDEX: dict[str, int] = {gt.value: i for i, gt in enumerate(GapType)}
ASSUMPTION_TYPE_TO_INDEX: dict[str, int] = {
    at.value: i for i, at in enumerate(AssumptionType)
}


@dataclass(frozen=True)
class MultitaskSample:
    """One annotated training sample for multi-task training."""

    text: str
    gap_type: str | None = None
    assumption_type: str | None = None


def load_multitask_annotations(path: str) -> list[MultitaskSample]:
    """Load annotated samples from a JSONL file.

    Each line: {"text": str, "gap_type"?: str, "assumption_type"?: str}
    Raises FileNotFoundError if path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Annotation JSONL not found: {path}")
    samples: list[MultitaskSample] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            samples.append(
                MultitaskSample(
                    text=rec["text"],
                    gap_type=rec.get("gap_type"),
                    assumption_type=rec.get("assumption_type"),
                )
            )
    return samples


def tokenize_sample(
    text: str,
    char_to_id: dict[str, int],
    bos_id: int,
    max_len: int,
) -> list[int]:
    """Character-level tokenization with BOS wrapping (microgpt.py convention).

    Returns list of integer token IDs: [BOS, char_ids..., BOS], truncated to
    max_len. Unknown characters map to bos_id.
    """
    ids = [bos_id]
    for ch in text:
        ids.append(char_to_id.get(ch, bos_id))
    ids.append(bos_id)
    return ids[:max_len]
