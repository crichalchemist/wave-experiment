---
id: ADR-028
title: Multi-task loss integration for DetectiveGPT
status: accepted
date: 2026-03-08
tags: [training, model, multi-task, loss, assumption-detection]
---

# ADR-028: Multi-task Loss Integration for DetectiveGPT

## Decision

Add a third output track (assumption type classification) to DetectiveGPT, implement `compute_multitask_loss()` for the three-head architecture, and provide a custom PyTorch training loop via `MultitaskTrainer`. The multi-task loss formula is:

```
L_total = L_language + ALPHA * L_gap + BETA * L_assumption
```

where `ALPHA = BETA = 0.3` (named constants in `src/core/model.py`).

## Context

DetectiveGPT had two output tracks (language model + gap detection) but assumption detection (Track 3) was listed as a stub. The A+B+C assumption taxonomy (Modules A/B/C) existed for runtime detection but had no representation in the model's forward pass or training pipeline.

Three approaches were considered:

1. **HuggingFace Trainer integration** — wrap DetectiveGPT in HF-compatible interface. Rejected: `forward()` returns a 3-tuple, incompatible with HF Trainer's single-loss assumption. Would require significant adapter code.
2. **Single shared encoder for all tracks** — shared gradients across all tasks. Rejected: gradient interference between LM, gap, and assumption objectives destabilizes training.
3. **Independent encoder pathways with detached backbone** — each detection track (gap, assumption) gets its own embedding + encoder + head, using `hidden.detach()` to sever gradient flow from the shared backbone. Chosen: prevents catastrophic interference, backbone trains only via L_language.

## Architecture

```
token_emb + pos_emb → backbone (TransformerEncoder)
     ↓                         ↓                              ↓
 lm_head              temporal_emb → temporal_encoder    assumption_emb → assumption_encoder
(vocab_size)           → gap_head (5 types)               → assumption_head (3 types)

hidden.detach() severs gradient flow from backbone to detection tracks.
```

### Per-token labels with document-level broadcast

Gap and assumption heads output `(batch, seq_len, N_TYPES)`. When only document-level labels are available, the label index is broadcast to every token position. Positions without labels use `IGNORE_INDEX = -100` (standard PyTorch convention for CrossEntropyLoss).

### Builder pattern

`build_multitask_trainer()` returns an un-started `MultitaskTrainer`, matching the codebase convention (`build_sft_trainer`, `build_dpo_trainer`).

### Training loop

```
for epoch in range(config.num_epochs):
    for batch in train_data:
        lm_logits, gap_logits, assumption_logits = model(batch["input_ids"])
        total, lm, gap, assume = compute_multitask_loss(...)
        optimizer.zero_grad()
        total.backward()
        clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
```

## Consequences

- `DetectiveGPT.forward()` now returns a 3-tuple instead of 2-tuple. This is a breaking change for any code calling forward() directly (only test_model.py was affected).
- The `data/annotations/` directory schema is now defined for multi-task JSONL: `{"text": str, "gap_type"?: str, "assumption_type"?: str}`.
- Synthetic data generator provides labeled training data from regex patterns aligned with Module A/B/C triggers.
- All new modules follow the existing torch fallback import pattern for CI environments without PyTorch.

## Files

### New
- `src/training/multitask_config.py` — `MultitaskTrainingConfig` frozen dataclass
- `src/training/multitask_loss.py` — `compute_multitask_loss()` function
- `src/training/multitask_dataset.py` — `MultitaskSample`, JSONL loader, tokenizer, label maps
- `src/training/generate_multitask_data.py` — synthetic data generator with regex classification
- `src/training/train_multitask.py` — `MultitaskTrainer` class + `build_multitask_trainer()` builder
- `tests/training/test_multitask_config.py` — config tests (4)
- `tests/training/test_multitask_loss.py` — loss function tests (3)
- `tests/training/test_multitask_dataset.py` — dataset loader tests (12)
- `tests/training/test_generate_multitask_data.py` — generator tests (9)
- `tests/training/test_train_multitask.py` — trainer tests (8)

### Modified
- `src/core/model.py` — added `N_ASSUMPTION_TYPES`, `BETA`, Track 3 layers, updated `forward()` to return 3-tuple
- `tests/core/test_model.py` — updated from 2-track to 3-track assertions
