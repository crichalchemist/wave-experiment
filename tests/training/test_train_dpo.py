"""Tests for DPO fine-tuning script."""
import json
from unittest.mock import MagicMock, patch
import pytest


def test_dpo_constants():
    from src.training.train_dpo import (
        DPO_OUTPUT_DIR, DPO_BETA, DPO_LR, DPO_EPOCHS, DPO_BATCH_SIZE,
        DPO_GRAD_ACCUM, LORA_RANK, LORA_ALPHA, DEFAULT_MODEL_ID,
    )
    assert DPO_OUTPUT_DIR == "checkpoints/dpo"
    assert DPO_BETA == 0.1
    assert DPO_LR == 5e-7
    assert DPO_EPOCHS == 1
    assert DPO_BATCH_SIZE == 1       # batch_size=1 for CPU training
    assert DPO_GRAD_ACCUM == 8       # effective batch = 8
    assert LORA_RANK == 16
    assert LORA_ALPHA == 32
    assert "DeepSeek" in DEFAULT_MODEL_ID


def test_load_preference_pairs_valid(tmp_path):
    from src.training.train_dpo import load_preference_pairs, PreferenceSample
    data = [
        {"instruction": "Find gaps", "rejected": "No gaps.", "chosen": "Temporal gap in 2013."},
        {"instruction": "Analyze", "rejected": "Nothing.", "chosen": "Evidential gap found."},
    ]
    f = tmp_path / "prefs.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in data))
    result = load_preference_pairs(str(f))
    assert len(result) == 2
    assert isinstance(result[0], PreferenceSample)
    assert result[0].instruction == "Find gaps"
    assert result[0].rejected == "No gaps."
    assert result[0].chosen == "Temporal gap in 2013."


def test_load_preference_pairs_missing_file():
    from src.training.train_dpo import load_preference_pairs
    with pytest.raises(FileNotFoundError):
        load_preference_pairs("/nonexistent/prefs.jsonl")


def test_load_preference_pairs_skips_blank_lines(tmp_path):
    from src.training.train_dpo import load_preference_pairs
    content = json.dumps({"instruction": "i", "rejected": "r", "chosen": "c"}) + "\n\n"
    f = tmp_path / "prefs.jsonl"
    f.write_text(content)
    result = load_preference_pairs(str(f))
    assert len(result) == 1


def test_preference_sample_is_immutable():
    from src.training.train_dpo import PreferenceSample
    s = PreferenceSample("i", "r", "c")
    with pytest.raises((AttributeError, TypeError)):
        s.instruction = "other"   # type: ignore


def test_preference_pairs_to_dataset():
    from src.training.train_dpo import PreferenceSample, preference_pairs_to_dataset
    samples = [
        PreferenceSample("prompt1", "bad1", "good1"),
        PreferenceSample("prompt2", "bad2", "good2"),
    ]
    ds = preference_pairs_to_dataset(samples)
    assert len(ds) == 2
    assert set(ds.column_names) == {"prompt", "chosen", "rejected"}
    assert ds[0]["prompt"] == "prompt1"
    assert ds[0]["chosen"] == "good1"
    assert ds[0]["rejected"] == "bad1"


def test_build_lora_config():
    from src.training.train_dpo import build_lora_config, LORA_RANK, LORA_ALPHA
    config = build_lora_config()
    assert config.r == LORA_RANK
    assert config.lora_alpha == LORA_ALPHA
    assert "q_proj" in config.target_modules
    assert "v_proj" in config.target_modules


def test_build_dpo_trainer_wires_config():
    """Verify DPOConfig receives the spec-mandated values."""
    captured: dict = {}

    class _CapturingConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    mock_trainer_cls = MagicMock(return_value=MagicMock())

    with (
        patch("src.training.train_dpo.DPOConfig", _CapturingConfig),
        patch("src.training.train_dpo.DPOTrainer", mock_trainer_cls),
    ):
        from src.training.train_dpo import build_dpo_trainer
        build_dpo_trainer(MagicMock(), MagicMock(), MagicMock())

    assert captured["output_dir"] == "checkpoints/dpo"
    assert captured["beta"] == 0.1
    assert captured["learning_rate"] == 5e-7
    assert captured["num_train_epochs"] == 1
    assert captured["per_device_train_batch_size"] == 1
    assert captured["gradient_accumulation_steps"] == 8
    assert captured["precompute_ref_log_probs"] is True
    assert captured["report_to"] == "none"
    assert mock_trainer_cls.called


def test_build_dpo_trainer_accepts_no_eval_dataset():
    """eval_dataset is optional — DPOTrainer should be called with None."""
    mock_config_cls = MagicMock(return_value=MagicMock())
    mock_trainer_cls = MagicMock(return_value=MagicMock())

    with (
        patch("src.training.train_dpo.DPOConfig", mock_config_cls),
        patch("src.training.train_dpo.DPOTrainer", mock_trainer_cls),
    ):
        from src.training.train_dpo import build_dpo_trainer
        build_dpo_trainer(MagicMock(), MagicMock(), MagicMock(), eval_dataset=None)

    call_kwargs = mock_trainer_cls.call_args.kwargs
    assert call_kwargs.get("eval_dataset") is None
