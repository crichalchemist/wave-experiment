"""Tests for GRPO fine-tuning script."""

from unittest.mock import MagicMock, patch


def test_gap_detection_reward_has_gap_keyword():
    from src.training.train_grpo import gap_detection_reward

    score = gap_detection_reward(
        "Find gaps",
        "There is a temporal gap in 2013 that was not documented in the records.",
    )
    assert score > 0.0


def test_gap_detection_reward_empty_completion():
    from src.training.train_grpo import gap_detection_reward

    score = gap_detection_reward("Find gaps", "")
    assert score == 0.0


def test_gap_detection_reward_returns_float():
    from src.training.train_grpo import gap_detection_reward

    result = gap_detection_reward("some prompt", "some completion")
    assert isinstance(result, float)


def test_gap_detection_reward_capped_at_one():
    from src.training.train_grpo import gap_detection_reward

    # Full-scoring completion: has gap keyword + gap type + long enough
    score = gap_detection_reward(
        "Analyze",
        "There is a significant temporal gap and evidential gap in the documentation for this period.",
    )
    assert 0.0 <= score <= 1.0


def test_gap_detection_reward_keyword_only():
    """Completion with a gap keyword but no named type and too short stays < 0.7."""
    from src.training.train_grpo import gap_detection_reward

    # "missing" is a gap indicator; no gap type named; short
    score = gap_detection_reward("p", "missing")
    assert score == 0.4


def test_gap_detection_reward_length_bonus():
    """Completion >= 50 chars that has keyword + type earns all three components."""
    from src.training.train_grpo import gap_detection_reward

    completion = "temporal gap is evidential in nature " + "x" * 20
    score = gap_detection_reward("p", completion)
    # keyword (0.4) + type (0.3) + length (0.3) = 1.0, capped at 1.0
    assert score == 1.0


def test_grpo_constants():
    from src.training.train_grpo import (
        GRPO_BATCH_SIZE,
        GRPO_EPOCHS,
        GRPO_LR,
        GRPO_OUTPUT_DIR,
    )

    assert GRPO_OUTPUT_DIR == "checkpoints/grpo"
    assert GRPO_LR == 1e-5
    assert GRPO_EPOCHS == 3
    assert GRPO_BATCH_SIZE == 4


def test_build_grpo_trainer_wires_config():
    """Verify GRPOConfig receives the spec-mandated values."""
    captured: dict = {}

    class _CapturingConfig:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    mock_trainer_cls = MagicMock(return_value=MagicMock())

    with (
        patch("src.training.train_grpo.GRPOConfig", _CapturingConfig),
        patch("src.training.train_grpo.GRPOTrainer", mock_trainer_cls),
    ):
        from src.training.train_grpo import build_grpo_trainer

        build_grpo_trainer(MagicMock(), MagicMock(), MagicMock())

    assert captured["output_dir"] == "checkpoints/grpo"
    assert captured["learning_rate"] == 1e-5
    assert captured["num_train_epochs"] == 3
    assert captured["per_device_train_batch_size"] == 4
    assert captured["report_to"] == "none"
    assert mock_trainer_cls.called


def test_build_grpo_trainer_uses_reward_fn():
    """Verify custom reward_fn is passed through to GRPOTrainer."""
    custom_reward = MagicMock(return_value=0.5)
    mock_config_cls = MagicMock(return_value=MagicMock())
    mock_trainer_cls = MagicMock(return_value=MagicMock())

    with (
        patch("src.training.train_grpo.GRPOConfig", mock_config_cls),
        patch("src.training.train_grpo.GRPOTrainer", mock_trainer_cls),
    ):
        from src.training.train_grpo import build_grpo_trainer

        build_grpo_trainer(MagicMock(), MagicMock(), MagicMock(), reward_fn=custom_reward)

    call_kwargs = mock_trainer_cls.call_args.kwargs
    assert call_kwargs.get("reward_funcs") is custom_reward


def test_build_grpo_trainer_default_reward_is_gap_detection():
    """When no reward_fn provided, GRPOTrainer receives gap_detection_reward."""
    from src.training.train_grpo import gap_detection_reward

    mock_config_cls = MagicMock(return_value=MagicMock())
    mock_trainer_cls = MagicMock(return_value=MagicMock())

    with (
        patch("src.training.train_grpo.GRPOConfig", mock_config_cls),
        patch("src.training.train_grpo.GRPOTrainer", mock_trainer_cls),
    ):
        from src.training.train_grpo import build_grpo_trainer

        build_grpo_trainer(MagicMock(), MagicMock(), MagicMock())

    call_kwargs = mock_trainer_cls.call_args.kwargs
    assert call_kwargs.get("reward_funcs") is gap_detection_reward
