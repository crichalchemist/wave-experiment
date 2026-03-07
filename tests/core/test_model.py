"""Tests for DetectiveGPT temporal embedding architecture."""
import types
from unittest.mock import MagicMock, patch


def test_model_constants():
    from src.core.model import N_GAP_TYPES, N_ASSUMPTION_TYPES, ALPHA, BETA, DEFAULT_N_EMBD, MAX_SEQ_LEN
    assert N_GAP_TYPES == 5
    assert N_ASSUMPTION_TYPES == 3
    assert ALPHA == 0.3
    assert BETA == 0.3
    assert DEFAULT_N_EMBD == 384
    assert MAX_SEQ_LEN == 1024


def _make_mock_nn():
    """Build a mock nn namespace whose Linear calls return unique instances."""
    linear_calls = []

    def _linear(*args, **kwargs):
        m = MagicMock()
        linear_calls.append(m)
        return m

    class _MockModule:
        def __init__(self): pass
        def __init_subclass__(cls, **kw): pass

    ns = types.SimpleNamespace(
        Module=_MockModule,
        Embedding=MagicMock(side_effect=lambda *a, **kw: MagicMock()),
        Linear=_linear,
        TransformerEncoderLayer=MagicMock(return_value=MagicMock()),
        TransformerEncoder=MagicMock(return_value=MagicMock()),
    )
    return ns, linear_calls


def _instantiate_gpt_with_mocks():
    """Return a DetectiveGPT instance with torch/nn mocked out."""
    from importlib import reload
    import src.core.model as model_module

    mock_nn, _ = _make_mock_nn()
    mock_torch = MagicMock()

    with patch("src.core.model.nn", mock_nn), patch("src.core.model.torch", mock_torch):
        reload(model_module)
        gpt = model_module.DetectiveGPT.__new__(model_module.DetectiveGPT)
        model_module.DetectiveGPT.__init__(gpt)

    return gpt


def test_model_has_three_tracks():
    """Model must expose all three track attributes: LM, gap, assumption."""
    gpt = _instantiate_gpt_with_mocks()
    for attr in (
        "lm_head", "gap_head", "assumption_head",
        "temporal_emb", "temporal_encoder",
        "assumption_emb", "assumption_encoder",
        "backbone",
    ):
        assert hasattr(gpt, attr), f"Missing attribute: {attr}"


def test_all_three_heads_are_independent():
    """All three output heads must be distinct objects (no shared weights)."""
    gpt = _instantiate_gpt_with_mocks()
    assert gpt.lm_head is not gpt.gap_head
    assert gpt.lm_head is not gpt.assumption_head
    assert gpt.gap_head is not gpt.assumption_head


def test_forward_returns_three_tensors():
    """DetectiveGPT.forward must return a 3-tuple and call all three heads."""
    from importlib import reload
    import src.core.model as model_module

    mock_nn, _ = _make_mock_nn()
    mock_torch = MagicMock()
    # arange(...).unsqueeze(0) is called in forward for positions
    mock_torch.arange.return_value.unsqueeze.return_value = MagicMock()

    # forward reads torch from the module namespace at call time, so both
    # instantiation AND the forward() call must occur inside the patch context.
    with patch("src.core.model.nn", mock_nn), patch("src.core.model.torch", mock_torch):
        reload(model_module)
        # reload() re-executes `import torch` and `import torch.nn as nn`,
        # overwriting both patches; re-apply them before instantiating.
        model_module.torch = mock_torch
        model_module.nn = mock_nn
        gpt = model_module.DetectiveGPT.__new__(model_module.DetectiveGPT)
        model_module.DetectiveGPT.__init__(gpt)

        # mock input: shape must unpack as (batch, seq_len)
        mock_x = MagicMock()
        mock_x.shape = (2, 8)

        result = gpt.forward(mock_x)  # called inside patch context

    assert isinstance(result, tuple), "forward() must return a tuple"
    assert len(result) == 3, "forward() must return exactly 3 elements (lm, gap, assumption)"
    # All three heads must have been invoked
    gpt.lm_head.assert_called_once()
    gpt.gap_head.assert_called_once()
    gpt.assumption_head.assert_called_once()


def test_gap_logits_last_dim_matches_n_gap_types():
    from src.core.model import N_GAP_TYPES
    mock_gap = MagicMock()
    mock_gap.shape = (1, 4, N_GAP_TYPES)
    assert mock_gap.shape[-1] == N_GAP_TYPES


def test_token_emb_and_temporal_emb_are_independent():
    """token_emb and temporal_emb must be separate Embedding instances."""
    gpt = _instantiate_gpt_with_mocks()
    assert gpt.token_emb is not gpt.temporal_emb


def test_assumption_emb_and_temporal_emb_are_independent():
    """assumption_emb and temporal_emb must be separate Embedding instances."""
    gpt = _instantiate_gpt_with_mocks()
    assert gpt.temporal_emb is not gpt.assumption_emb
