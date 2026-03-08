"""Tests for MultitaskTrainer and builder function."""
import pytest
from unittest.mock import MagicMock


def test_train_step_result_is_frozen():
    from src.training.train_multitask import TrainStepResult
    r = TrainStepResult(
        step=1, total_loss=0.5, lm_loss=0.3,
        gap_loss=0.1, assumption_loss=0.1, duration_ms=50,
    )
    with pytest.raises((AttributeError, TypeError)):
        r.step = 2  # type: ignore[misc]


def test_train_step_result_fields():
    from src.training.train_multitask import TrainStepResult
    r = TrainStepResult(
        step=42, total_loss=1.5, lm_loss=0.8,
        gap_loss=0.4, assumption_loss=0.3, duration_ms=120,
    )
    assert r.step == 42
    assert r.total_loss == 1.5
    assert r.lm_loss == 0.8
    assert r.gap_loss == 0.4
    assert r.assumption_loss == 0.3
    assert r.duration_ms == 120


def test_build_multitask_trainer_returns_trainer():
    from src.training.train_multitask import build_multitask_trainer, MultitaskTrainer
    trainer = build_multitask_trainer(MagicMock(), train_data=[])
    assert isinstance(trainer, MultitaskTrainer)


def test_build_multitask_trainer_default_config():
    from src.training.train_multitask import build_multitask_trainer
    trainer = build_multitask_trainer(MagicMock(), train_data=[])
    assert trainer.config.alpha == 0.3
    assert trainer.config.beta == 0.3
    assert trainer.config.num_epochs == 10
    assert trainer.config.learning_rate == 1e-4


def test_build_multitask_trainer_custom_config():
    from src.training.train_multitask import build_multitask_trainer
    from src.training.multitask_config import MultitaskTrainingConfig
    cfg = MultitaskTrainingConfig(alpha=0.5, beta=0.1, num_epochs=3)
    trainer = build_multitask_trainer(MagicMock(), train_data=[], config=cfg)
    assert trainer.config.alpha == 0.5
    assert trainer.config.beta == 0.1
    assert trainer.config.num_epochs == 3


def test_trainer_does_not_start_on_build():
    from src.training.train_multitask import build_multitask_trainer
    trainer = build_multitask_trainer(MagicMock(), train_data=[])
    assert trainer.step_history == []


def test_trainer_train_empty_data():
    """Training with empty data should return empty history."""
    from src.training.train_multitask import build_multitask_trainer
    trainer = build_multitask_trainer(MagicMock(), train_data=[])
    result = trainer.train()
    assert result == []
    assert trainer.step_history == []


def test_trainer_exposes_model():
    from src.training.train_multitask import build_multitask_trainer
    model = MagicMock()
    trainer = build_multitask_trainer(model, train_data=[])
    assert trainer.model is model
