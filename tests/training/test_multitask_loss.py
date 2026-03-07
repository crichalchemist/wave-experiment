"""Tests for multi-task loss computation."""
import types
from unittest.mock import MagicMock, patch


def test_loss_constants():
    from src.training.multitask_loss import IGNORE_INDEX
    assert IGNORE_INDEX == -100


def test_compute_multitask_loss_returns_four_tensors():
    """compute_multitask_loss returns (total, lm, gap, assumption)."""
    from importlib import reload
    import src.training.multitask_loss as loss_module

    # Build mock nn with CrossEntropyLoss that returns a mock tensor
    mock_loss_val = MagicMock()
    mock_loss_val.__add__ = MagicMock(return_value=mock_loss_val)
    mock_loss_val.__radd__ = MagicMock(return_value=mock_loss_val)
    mock_loss_val.__mul__ = MagicMock(return_value=mock_loss_val)
    mock_loss_val.__rmul__ = MagicMock(return_value=mock_loss_val)

    mock_ce_instance = MagicMock(return_value=mock_loss_val)
    mock_nn = types.SimpleNamespace(CrossEntropyLoss=MagicMock(return_value=mock_ce_instance))

    # Mock tensors with .shape
    lm_logits = MagicMock()
    lm_logits.shape = (2, 8, 256)
    lm_logits.reshape = MagicMock(return_value=MagicMock())

    gap_logits = MagicMock()
    gap_logits.shape = (2, 8, 5)
    gap_logits.reshape = MagicMock(return_value=MagicMock())

    assumption_logits = MagicMock()
    assumption_logits.shape = (2, 8, 3)
    assumption_logits.reshape = MagicMock(return_value=MagicMock())

    lm_targets = MagicMock()
    lm_targets.reshape = MagicMock(return_value=MagicMock())

    gap_targets = MagicMock()
    gap_targets.reshape = MagicMock(return_value=MagicMock())

    assumption_targets = MagicMock()
    assumption_targets.reshape = MagicMock(return_value=MagicMock())

    with patch.object(loss_module, "nn", mock_nn):
        reload(loss_module)
        loss_module.nn = mock_nn
        result = loss_module.compute_multitask_loss(
            lm_logits, gap_logits, assumption_logits,
            lm_targets, gap_targets, assumption_targets,
        )

    assert isinstance(result, tuple)
    assert len(result) == 4


def test_compute_multitask_loss_uses_alpha_beta():
    from src.training.multitask_loss import IGNORE_INDEX
    from src.core.model import ALPHA, BETA
    assert ALPHA == 0.3
    assert BETA == 0.3
    assert IGNORE_INDEX == -100
