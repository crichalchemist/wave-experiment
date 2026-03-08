"""Custom PyTorch training loop for multi-task DetectiveGPT.

DetectiveGPT's forward() returns a 3-tuple (lm_logits, gap_logits,
assumption_logits) which is incompatible with HuggingFace Trainer's
single-loss assumption. This module provides a custom training loop
computing L_total = L_language + alpha * L_gap + beta * L_assumption.

Usage:
    trainer = build_multitask_trainer(model, train_data=samples)
    results = trainer.train()
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import torch
except ImportError:
    _current = sys.modules.get(__name__)
    if getattr(_current, "torch", None) is None:
        torch = None  # type: ignore[assignment]

from src.training.multitask_config import MultitaskTrainingConfig
from src.training.multitask_loss import IGNORE_INDEX, compute_multitask_loss
from src.training.multitask_dataset import (
    GAP_TYPE_TO_INDEX,
    ASSUMPTION_TYPE_TO_INDEX,
    MultitaskSample,
    tokenize_sample,
)


@dataclass(frozen=True)
class TrainStepResult:
    """Immutable record of a single training step."""

    step: int
    total_loss: float
    lm_loss: float
    gap_loss: float
    assumption_loss: float
    duration_ms: int


class MultitaskTrainer:
    """Custom PyTorch trainer for three-head DetectiveGPT.

    Constructed via build_multitask_trainer(). Call .train() to run.
    """

    def __init__(
        self,
        model: object,
        config: MultitaskTrainingConfig,
        train_data: list[MultitaskSample],
    ) -> None:
        self.model = model
        self.config = config
        self.train_data = train_data
        self.step_history: list[TrainStepResult] = []

    def _build_char_vocab(self) -> tuple[dict[str, int], int]:
        """Build character-to-id mapping from training data."""
        chars: set[str] = set()
        for sample in self.train_data:
            chars.update(sample.text)
        char_to_id = {ch: i for i, ch in enumerate(sorted(chars))}
        bos_id = len(char_to_id)
        return char_to_id, bos_id

    def _prepare_batch(
        self,
        samples: list[MultitaskSample],
        char_to_id: dict[str, int],
        bos_id: int,
    ) -> "dict[str, torch.Tensor]":
        """Tokenize samples and build input/target tensors."""
        _t = sys.modules[__name__].torch  # type: ignore[attr-defined]
        max_len = self.config.max_seq_len

        input_ids_list = []
        lm_targets_list = []
        gap_targets_list = []
        assumption_targets_list = []

        for sample in samples:
            tokens = tokenize_sample(sample.text, char_to_id, bos_id, max_len)
            # LM targets: shifted by 1 (predict next token)
            lm_tgt = tokens[1:] + [bos_id]

            # Gap target: broadcast document-level label to all positions
            if sample.gap_type is not None and sample.gap_type in GAP_TYPE_TO_INDEX:
                gap_label = GAP_TYPE_TO_INDEX[sample.gap_type]
                gap_tgt = [gap_label] * len(tokens)
            else:
                gap_tgt = [IGNORE_INDEX] * len(tokens)

            # Assumption target: broadcast document-level label to all positions
            if (
                sample.assumption_type is not None
                and sample.assumption_type in ASSUMPTION_TYPE_TO_INDEX
            ):
                assume_label = ASSUMPTION_TYPE_TO_INDEX[sample.assumption_type]
                assume_tgt = [assume_label] * len(tokens)
            else:
                assume_tgt = [IGNORE_INDEX] * len(tokens)

            # Pad to max_len
            pad_len = max_len - len(tokens)
            tokens += [bos_id] * pad_len
            lm_tgt += [IGNORE_INDEX] * pad_len
            gap_tgt += [IGNORE_INDEX] * pad_len
            assume_tgt += [IGNORE_INDEX] * pad_len

            input_ids_list.append(tokens)
            lm_targets_list.append(lm_tgt)
            gap_targets_list.append(gap_tgt)
            assumption_targets_list.append(assume_tgt)

        return {
            "input_ids": _t.tensor(input_ids_list, dtype=_t.long),
            "lm_targets": _t.tensor(lm_targets_list, dtype=_t.long),
            "gap_targets": _t.tensor(gap_targets_list, dtype=_t.long),
            "assumption_targets": _t.tensor(assumption_targets_list, dtype=_t.long),
        }

    def _save_checkpoint(self, step: int) -> None:
        """Save model state_dict to config.output_dir/step-N/model.pt."""
        _t = sys.modules[__name__].torch  # type: ignore[attr-defined]
        out_dir = Path(self.config.output_dir) / f"step-{step}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _t.save(self.model.state_dict(), out_dir / "model.pt")

    def train(self) -> list[TrainStepResult]:
        """Run the multi-task training loop.

        Returns list of TrainStepResult for logged steps.
        """
        if not self.train_data:
            return []

        _t = sys.modules[__name__].torch  # type: ignore[attr-defined]
        char_to_id, bos_id = self._build_char_vocab()

        optimizer = _t.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        global_step = 0
        batch_size = self.config.batch_size

        for epoch in range(self.config.num_epochs):
            for i in range(0, len(self.train_data), batch_size):
                batch_samples = self.train_data[i : i + batch_size]
                batch = self._prepare_batch(batch_samples, char_to_id, bos_id)

                t0 = time.monotonic()

                lm_logits, gap_logits, assumption_logits = self.model(
                    batch["input_ids"]
                )
                total, lm_loss, gap_loss, assumption_loss = compute_multitask_loss(
                    lm_logits,
                    gap_logits,
                    assumption_logits,
                    batch["lm_targets"],
                    batch["gap_targets"],
                    batch["assumption_targets"],
                    alpha=self.config.alpha,
                    beta=self.config.beta,
                )

                optimizer.zero_grad()
                total.backward()
                _t.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                optimizer.step()

                global_step += 1
                duration_ms = int((time.monotonic() - t0) * 1000)

                if global_step % self.config.log_interval == 0:
                    result = TrainStepResult(
                        step=global_step,
                        total_loss=float(total.item()),
                        lm_loss=float(lm_loss.item()),
                        gap_loss=float(gap_loss.item()),
                        assumption_loss=float(assumption_loss.item()),
                        duration_ms=duration_ms,
                    )
                    self.step_history.append(result)

                if global_step % self.config.save_interval == 0:
                    self._save_checkpoint(global_step)

        # Final checkpoint
        if global_step > 0:
            self._save_checkpoint(global_step)

        return self.step_history


def build_multitask_trainer(
    model: object,
    train_data: list[MultitaskSample] | list[object] | None = None,
    config: MultitaskTrainingConfig | None = None,
) -> MultitaskTrainer:
    """Build a MultitaskTrainer (un-started).

    Caller controls when .train() is invoked, matching the build_sft_trainer
    and build_dpo_trainer convention.
    """
    if config is None:
        config = MultitaskTrainingConfig()
    if train_data is None:
        train_data = []
    return MultitaskTrainer(
        model=model,
        config=config,
        train_data=train_data,  # type: ignore[arg-type]
    )
