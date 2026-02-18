"""Tests for SFT warm-up trainer and gap annotation loader."""

import json
from unittest.mock import MagicMock, patch

import pytest


def test_build_sft_trainer_returns_trainer():
    from src.training.train_sft import build_sft_trainer
    from transformers import Trainer

    model = MagicMock()
    tokenizer = MagicMock()
    train_ds = MagicMock()
    eval_ds = MagicMock()
    with patch("transformers.Trainer.__init__", return_value=None):
        result = build_sft_trainer(model, tokenizer, train_ds, eval_ds)
    # Patched __init__ returns None, so isinstance check is not reliable;
    # verify the return value is a Trainer subclass instance.
    assert isinstance(result, Trainer)


def test_training_args_config():
    from src.training.train_sft import (
        SFT_EPOCHS,
        SFT_LR,
        SFT_OUTPUT_DIR,
    )

    assert SFT_EPOCHS == 3
    assert SFT_LR == 2e-5
    assert SFT_OUTPUT_DIR == "checkpoints/sft"


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
