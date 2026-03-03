"""
HybridGraphLayer: LLM-scored edge weights fed into Bi-GAT (GATv2Conv).

Two-step reasoning:
  1. LLM scores edge plausibility: P(A→B | context) as a float in [0, 1]
  2. Edge weights feed GATv2Conv for structural aggregation

The LLM scorer is injected (ModelProvider) so it works without a live model in tests.
"""
from __future__ import annotations

import sys
import warnings

try:
    import torch

    # Suppress upstream deprecation warnings from torch_geometric internals:
    # - torch_geometric.distributed (deprecated in 2.7.0, imported transitively)
    # - torch.jit.script (used by torch_geometric.nn.pool for TypedDict JIT)
    # Neither API is used by our code; these fire on import of GATv2Conv.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*torch_geometric\.distributed.*deprecated")
        warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*deprecated")
        from torch_geometric.nn import GATv2Conv

    import torch.nn as nn
except ImportError:
    _current = sys.modules.get(__name__)
    if getattr(_current, "torch", None) is None:
        torch = None  # type: ignore[assignment]
    if getattr(_current, "nn", None) is None:
        nn = None  # type: ignore[assignment]
    if getattr(_current, "GATv2Conv", None) is None:
        GATv2Conv = None  # type: ignore[assignment]

from src.core.providers import ModelProvider

# Named constants
EDGE_SCORE_PROMPT: str = (
    "Score the plausibility of this relationship on a scale from 0.0 to 1.0. "
    "Return only a single float.\n\nRelationship: {source} → {target}\nContext: {context}"
)
DEFAULT_HIDDEN_DIM: int = 64
DEFAULT_HEADS: int = 4
_FALLBACK_EDGE_SCORE: float = 0.5  # used when LLM response cannot be parsed


class _SentinelModule:
    """Fallback base when torch is unavailable."""

    def __init__(self) -> None:
        pass

    def __init_subclass__(cls, **kw: object) -> None:
        pass


def score_edge(
    source: str,
    target: str,
    context: str,
    provider: ModelProvider,
) -> float:
    """
    Ask the LLM to score edge plausibility P(source→target | context).
    Returns a float in [0.0, 1.0]. Falls back to _FALLBACK_EDGE_SCORE if
    the response cannot be parsed as a float.
    """
    prompt = EDGE_SCORE_PROMPT.format(source=source, target=target, context=context)
    response = provider.complete(prompt)
    try:
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        return _FALLBACK_EDGE_SCORE


class HybridGraphLayer(_SentinelModule):
    """
    Bi-directional GAT layer with LLM-scored edge weights.

    Usage:
        layer = HybridGraphLayer(in_dim=64, out_dim=64)
        out = layer(x, edge_index, edge_weights)
        # out embeddings differ from x because attention is weighted by LLM scores
    """

    def __init__(
        self,
        in_dim: int = DEFAULT_HIDDEN_DIM,
        out_dim: int = DEFAULT_HIDDEN_DIM,
        heads: int = DEFAULT_HEADS,
    ) -> None:
        _n = sys.modules[__name__].nn
        _gat_cls = sys.modules[__name__].GATv2Conv
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # GATv2Conv with edge_dim=1 to accept scalar edge weights
        self.gat = _gat_cls(in_dim, out_dim // heads, heads=heads, edge_dim=1)

    def forward(
        self,
        x: "torch.Tensor",
        edge_index: "torch.Tensor",
        edge_weights: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Args:
            x: (num_nodes, in_dim) node feature matrix
            edge_index: (2, num_edges) COO edge indices
            edge_weights: (num_edges, 1) LLM-scored plausibility weights
        Returns:
            (num_nodes, out_dim) updated node embeddings
        """
        return self.gat(x, edge_index, edge_attr=edge_weights)
