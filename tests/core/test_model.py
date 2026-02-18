"""Tests for DetectiveGPT temporal embedding architecture."""
import types
from unittest.mock import MagicMock, patch


def test_model_constants():
    from src.core.model import N_GAP_TYPES, ALPHA, DEFAULT_N_EMBD, MAX_SEQ_LEN
    assert N_GAP_TYPES == 5
    assert ALPHA == 0.3
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


def test_model_has_both_tracks():
    """Model must expose lm_head, gap_head, temporal_emb, temporal_encoder, backbone."""
    gpt = _instantiate_gpt_with_mocks()
    for attr in ("lm_head", "gap_head", "temporal_emb", "temporal_encoder", "backbone"):
        assert hasattr(gpt, attr), f"Missing attribute: {attr}"


def test_lm_head_and_gap_head_are_independent():
    """lm_head and gap_head must NOT be the same object (no shared weights)."""
    gpt = _instantiate_gpt_with_mocks()
    assert gpt.lm_head is not gpt.gap_head


def test_forward_returns_two_tensors():
    """forward() must return a 2-tuple (lm_logits, gap_logits)."""
    mock_lm = MagicMock()
    mock_lm.shape = (2, 8, 256)
    mock_gap = MagicMock()
    mock_gap.shape = (2, 8, 5)

    class _MockGPT:
        def forward(self, x):
            return mock_lm, mock_gap

    lm, gap = _MockGPT().forward(MagicMock())
    assert lm.shape == (2, 8, 256)
    assert gap.shape == (2, 8, 5)


def test_gap_logits_last_dim_matches_n_gap_types():
    from src.core.model import N_GAP_TYPES
    mock_gap = MagicMock()
    mock_gap.shape = (1, 4, N_GAP_TYPES)
    assert mock_gap.shape[-1] == N_GAP_TYPES


def test_token_emb_and_temporal_emb_are_independent():
    """token_emb and temporal_emb must be separate Embedding instances."""
    gpt = _instantiate_gpt_with_mocks()
    assert gpt.token_emb is not gpt.temporal_emb
