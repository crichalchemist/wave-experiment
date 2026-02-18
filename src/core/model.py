"""
DetectiveGPT: multi-task GPT with separate temporal embedding and gap detection tracks.

Architecture:
  token_emb + pos_emb → backbone (TransformerEncoder)
       ↓                         ↓
   lm_head              temporal_emb → temporal_encoder → gap_head
  (language model)                   (gap type classification)

The two output tracks are weight-independent. Multi-task loss:
  L_total = L_language + ALPHA * L_gap
"""
from __future__ import annotations

import sys

try:
    import torch
    import torch.nn as nn
except ImportError:
    # Only reset to None if not already injected (e.g. by a test mock via patch + reload).
    # importlib.reload() re-executes in the existing namespace, so the patched value is
    # visible via sys.modules.get(__name__) — the guard lets mocks survive reload().
    _current = sys.modules.get(__name__)
    _current_torch = getattr(_current, "torch", None) if _current is not None else None
    _current_nn = getattr(_current, "nn", None) if _current is not None else None
    if _current_torch is None:
        torch = None  # type: ignore[assignment]
    if _current_nn is None:
        nn = None  # type: ignore[assignment]

# Named constants
N_GAP_TYPES: int = 5  # temporal, evidential, contradiction, normative, doctrinal
ALPHA: float = 0.3  # gap loss weight in multi-task objective
DEFAULT_N_LAYER: int = 6
DEFAULT_N_HEAD: int = 4
DEFAULT_N_EMBD: int = 384
DEFAULT_VOCAB_SIZE: int = 256
MAX_SEQ_LEN: int = 1024


class _SentinelModule:
    """Fallback base used when torch is unavailable; keeps class definition valid."""

    def __init__(self) -> None:
        pass

    def __init_subclass__(cls, **kwargs: object) -> None:
        pass


class DetectiveGPT(_SentinelModule):
    """
    Multi-task GPT with independent language model and gap detection tracks.

    forward() returns (lm_logits, gap_logits):
      lm_logits:  (batch, seq_len, vocab_size)
      gap_logits: (batch, seq_len, N_GAP_TYPES)
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        n_layer: int = DEFAULT_N_LAYER,
        n_head: int = DEFAULT_N_HEAD,
        n_embd: int = DEFAULT_N_EMBD,
    ) -> None:
        # Read nn from the module's live namespace so test mocks injected via
        # patch("src.core.model.nn", ...) are visible at instantiation time.
        _n = sys.modules[__name__].nn  # type: ignore[attr-defined]

        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        # Shared token + position embeddings
        self.token_emb = _n.Embedding(vocab_size, n_embd)
        self.pos_emb = _n.Embedding(MAX_SEQ_LEN, n_embd)

        # Shared transformer backbone
        encoder_layer = _n.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            batch_first=True,
        )
        self.backbone = _n.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # Track 1: language model head (independent weights)
        self.lm_head = _n.Linear(n_embd, vocab_size, bias=False)

        # Track 2: temporal gap detection (entirely separate weights)
        self.temporal_emb = _n.Embedding(MAX_SEQ_LEN, n_embd)
        self.temporal_encoder = _n.Linear(n_embd * 2, n_embd)
        self.gap_head = _n.Linear(n_embd, N_GAP_TYPES, bias=False)

    def forward(
        self, x: "torch.Tensor"
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Args:
            x: (batch, seq_len) integer token ids
        Returns:
            (lm_logits, gap_logits)
        """
        _t = sys.modules[__name__].torch  # type: ignore[attr-defined]
        batch, seq_len = x.shape
        positions = _t.arange(seq_len, device=x.device).unsqueeze(0)

        # Shared backbone
        tok = self.token_emb(x) + self.pos_emb(positions)
        hidden = self.backbone(tok)

        # Track 1: language model logits
        lm_logits = self.lm_head(hidden)

        # Track 2: temporal gap detection (detach prevents gradient flow into backbone)
        temporal = self.temporal_emb(positions).expand(batch, -1, -1)
        fused = self.temporal_encoder(_t.cat([hidden.detach(), temporal], dim=-1))
        gap_logits = self.gap_head(fused)

        return lm_logits, gap_logits
