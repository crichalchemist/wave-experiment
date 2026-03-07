"""Tests for multi-task training configuration."""
import pytest


def test_config_constants():
    from src.training.multitask_config import (
        MULTITASK_OUTPUT_DIR, MULTITASK_LR, MULTITASK_EPOCHS, MULTITASK_BATCH_SIZE,
    )
    assert MULTITASK_OUTPUT_DIR == "checkpoints/multitask"
    assert MULTITASK_LR == 1e-4
    assert MULTITASK_EPOCHS == 10
    assert MULTITASK_BATCH_SIZE == 4


def test_config_is_frozen():
    from src.training.multitask_config import MultitaskTrainingConfig
    cfg = MultitaskTrainingConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.learning_rate = 0.1  # type: ignore[misc]


def test_config_defaults():
    from src.training.multitask_config import MultitaskTrainingConfig
    cfg = MultitaskTrainingConfig()
    assert cfg.alpha == 0.3
    assert cfg.beta == 0.3
    assert cfg.output_dir == "checkpoints/multitask"
    assert cfg.learning_rate == 1e-4
    assert cfg.num_epochs == 10
    assert cfg.batch_size == 4
    assert cfg.grad_clip == 1.0


def test_config_custom_values():
    from src.training.multitask_config import MultitaskTrainingConfig
    cfg = MultitaskTrainingConfig(alpha=0.5, beta=0.2, num_epochs=20)
    assert cfg.alpha == 0.5
    assert cfg.beta == 0.2
    assert cfg.num_epochs == 20
