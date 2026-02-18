"""SFT warm-up trainer using HuggingFace Trainer API.

Provides a minimal supervised fine-tuning loop configured for gap-annotation
data. build_sft_trainer constructs (but does not start) the training run so
callers retain full control over when .train() is invoked.
"""

from __future__ import annotations

import datasets
from transformers import Trainer, TrainingArguments

SFT_OUTPUT_DIR: str = "checkpoints/sft"
SFT_EPOCHS: int = 3
SFT_LR: float = 2e-5
SFT_BATCH_SIZE: int = 8
SFT_EVAL_STEPS: int = 100


def build_sft_trainer(
    model: object,
    tokenizer: object,
    train_dataset: object,
    eval_dataset: object,
) -> Trainer:
    """Build a HuggingFace Trainer for SFT warm-up.

    3 epochs, lr=2e-5, saves to checkpoints/sft.  The trainer is returned
    un-started so callers decide when to invoke .train().
    """
    args = TrainingArguments(
        output_dir=SFT_OUTPUT_DIR,
        num_train_epochs=SFT_EPOCHS,
        learning_rate=SFT_LR,
        per_device_train_batch_size=SFT_BATCH_SIZE,
        eval_strategy="steps",  # renamed from evaluation_strategy in transformers 5.x
        eval_steps=SFT_EVAL_STEPS,
        save_strategy="steps",
        save_steps=SFT_EVAL_STEPS,
        load_best_model_at_end=True,
        report_to="none",  # disable wandb/tensorboard
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def load_gap_annotations(path: str) -> datasets.Dataset:
    """Load gap annotation JSONL from *path* and return the train split.

    Each record must contain: text (str), gap_type (str), label (int 0/1).
    Raises FileNotFoundError (via datasets) when path does not exist.
    """
    ds = datasets.load_dataset("json", data_files=path, split="train")
    return ds
