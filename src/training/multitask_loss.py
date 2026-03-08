"""Multi-task loss computation for DetectiveGPT.

L_total = L_language + alpha * L_gap + beta * L_assumption
"""

from __future__ import annotations

import sys

try:
    import torch
    import torch.nn as nn
except ImportError:
    _current = sys.modules.get(__name__)
    if getattr(_current, "torch", None) is None:
        torch = None  # type: ignore[assignment]
    if getattr(_current, "nn", None) is None:
        nn = None  # type: ignore[assignment]

from src.core.model import ALPHA, BETA

IGNORE_INDEX: int = -100


def compute_multitask_loss(
    lm_logits: "torch.Tensor",
    gap_logits: "torch.Tensor",
    assumption_logits: "torch.Tensor",
    lm_targets: "torch.Tensor",
    gap_targets: "torch.Tensor",
    assumption_targets: "torch.Tensor",
    alpha: float = ALPHA,
    beta: float = BETA,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Compute combined multi-task loss.

    Returns:
        (total_loss, lm_loss, gap_loss, assumption_loss)
    """
    _nn = sys.modules[__name__].nn  # type: ignore[attr-defined]
    ce = _nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    b, s, v = lm_logits.shape
    lm_loss = ce(lm_logits.reshape(b * s, v), lm_targets.reshape(b * s))

    _, _, g = gap_logits.shape
    gap_loss = ce(gap_logits.reshape(b * s, g), gap_targets.reshape(b * s))

    _, _, a = assumption_logits.shape
    assumption_loss = ce(
        assumption_logits.reshape(b * s, a), assumption_targets.reshape(b * s)
    )

    total = lm_loss + alpha * gap_loss + beta * assumption_loss
    return total, lm_loss, gap_loss, assumption_loss
