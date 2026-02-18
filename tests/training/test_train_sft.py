"""Tests for SFT warm-up trainer and gap annotation loader."""

import json
from unittest.mock import MagicMock, patch

import pytest


def test_build_sft_trainer_wires_args():
    """Verify TrainingArguments receives the spec-mandated config values.

    Trainer requires PyTorch (not available in CI / 3.14 dev env), so both
    TrainingArguments and Trainer are mocked at the module level.  The test
    captures what kwargs were passed to TrainingArguments to verify wiring.
    """
    from src.training.train_sft import build_sft_trainer

    captured: dict = {}

    class _CapturingArgs:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    mock_trainer_cls = MagicMock(return_value=MagicMock())

    with (
        patch("src.training.train_sft.TrainingArguments", _CapturingArgs),
        patch("src.training.train_sft.Trainer", mock_trainer_cls),
    ):
        build_sft_trainer(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    assert captured["output_dir"] == "checkpoints/sft"
    assert captured["num_train_epochs"] == 3
    assert captured["learning_rate"] == 2e-5
    assert captured["report_to"] == "none"
    assert mock_trainer_cls.called


def test_training_args_config():
    from src.training.train_sft import (
        SFT_BATCH_SIZE,
        SFT_EPOCHS,
        SFT_EVAL_STEPS,
        SFT_LR,
        SFT_OUTPUT_DIR,
    )

    assert SFT_EPOCHS == 3
    assert SFT_LR == 2e-5
    assert SFT_OUTPUT_DIR == "checkpoints/sft"
    assert SFT_BATCH_SIZE == 8
    assert SFT_EVAL_STEPS == 100


def test_load_gap_annotations_invalid_path():
    from src.training.train_sft import load_gap_annotations

    with pytest.raises(Exception):  # FileNotFoundError or datasets error
        load_gap_annotations("/nonexistent/path.jsonl")


def test_load_gap_annotations_valid_data(tmp_path):
    from src.training.train_sft import load_gap_annotations

    data = [
        {"text": "Entity A was active in 2013.", "gap_type": "temporal", "label": 1},
        {"text": "No gaps here.", "gap_type": "none", "label": 0},
    ]
    jsonl_file = tmp_path / "annotations.jsonl"
    jsonl_file.write_text("\n".join(json.dumps(r) for r in data))
    dataset = load_gap_annotations(str(jsonl_file))
    assert len(dataset) == 2
    assert "text" in dataset.column_names
    assert "label" in dataset.column_names
