"""Configuration for multi-task training of DetectiveGPT."""

from __future__ import annotations

from dataclasses import dataclass

MULTITASK_OUTPUT_DIR: str = "checkpoints/multitask"
MULTITASK_LR: float = 1e-4
MULTITASK_EPOCHS: int = 10
MULTITASK_BATCH_SIZE: int = 4
MULTITASK_LOG_INTERVAL: int = 10
MULTITASK_SAVE_INTERVAL: int = 100
MULTITASK_MAX_SEQ_LEN: int = 256
MULTITASK_GRAD_CLIP: float = 1.0


@dataclass(frozen=True)
class MultitaskTrainingConfig:
    """Immutable configuration for multi-task training runs."""

    output_dir: str = MULTITASK_OUTPUT_DIR
    learning_rate: float = MULTITASK_LR
    num_epochs: int = MULTITASK_EPOCHS
    batch_size: int = MULTITASK_BATCH_SIZE
    alpha: float = 0.3
    beta: float = 0.3
    log_interval: int = MULTITASK_LOG_INTERVAL
    save_interval: int = MULTITASK_SAVE_INTERVAL
    max_seq_len: int = MULTITASK_MAX_SEQ_LEN
    grad_clip: float = MULTITASK_GRAD_CLIP
