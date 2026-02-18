"""GRPO fine-tuning for gap detection quality.

GRPOTrainer optimizes the model using group relative policy optimization.
The reward function is the specification of what 'better gap detection' means.
"""

from __future__ import annotations

from typing import Callable

# Module-level import with fallback so patch("src.training.train_grpo.GRPOConfig", ...)
# and patch("src.training.train_grpo.GRPOTrainer", ...) bind correctly in tests
# while avoiding ImportError when torch/trl are absent in the dev env.
try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:  # torch not available in dev env
    GRPOConfig = None  # type: ignore[assignment,misc]
    GRPOTrainer = None  # type: ignore[assignment,misc]

# Named constants — no magic numbers
GRPO_OUTPUT_DIR: str = "checkpoints/grpo"
GRPO_LR: float = 1e-5
GRPO_EPOCHS: int = 3
GRPO_BATCH_SIZE: int = 4

# Gap detection keywords — completions mentioning these score higher
_GAP_INDICATORS: frozenset[str] = frozenset(
    {
        "gap",
        "missing",
        "absent",
        "undocumented",
        "unaccounted",
        "temporal",
        "evidential",
        "contradiction",
        "normative",
        "doctrinal",
    }
)

# Named gap types used for the +0.3 type-identification bonus
_GAP_TYPES: frozenset[str] = frozenset(
    {"temporal", "evidential", "contradiction", "normative", "doctrinal"}
)

# Minimum completion length to earn the substantive-content bonus
_MIN_SUBSTANTIVE_LENGTH: int = 50

# Scoring weights — kept as constants so they are auditable and testable
_SCORE_KEYWORD_PRESENT: float = 0.4
_SCORE_TYPE_NAMED: float = 0.3
_SCORE_SUBSTANTIVE: float = 0.3
_SCORE_MAX: float = 1.0


def gap_detection_reward(prompt: str, completion: str) -> float:
    """Score a (prompt, completion) pair on gap detection quality.

    Scoring logic encodes the operational definition of 'better gap detection':
    a response that names a gap, identifies its type, and explains it at length
    is more useful than a terse or vague one.

    Returns a float in [0.0, 1.0].
    """
    completion_lower = completion.lower()
    score = 0.0

    # Presence of any gap-indicator keyword
    if any(kw in completion_lower for kw in _GAP_INDICATORS):
        score += _SCORE_KEYWORD_PRESENT

    # Explicit naming of a recognised gap type
    if any(gt in completion_lower for gt in _GAP_TYPES):
        score += _SCORE_TYPE_NAMED

    # Substantive length signals a complete explanation rather than a fragment
    if len(completion) >= _MIN_SUBSTANTIVE_LENGTH:
        score += _SCORE_SUBSTANTIVE

    return min(score, _SCORE_MAX)


def build_grpo_trainer(
    model: object,
    tokenizer: object,
    train_dataset: object,
    reward_fn: Callable | None = None,
) -> object:
    """Build a trl GRPOTrainer for gap detection fine-tuning.

    Constructs but does not invoke .train() — callers decide when to start
    the run.  reward_fn defaults to gap_detection_reward so the optimisation
    target is always explicit.
    """
    config = GRPOConfig(
        output_dir=GRPO_OUTPUT_DIR,
        learning_rate=GRPO_LR,
        num_train_epochs=GRPO_EPOCHS,
        per_device_train_batch_size=GRPO_BATCH_SIZE,
        report_to="none",
    )
    reward = reward_fn or gap_detection_reward
    return GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        reward_funcs=reward,
    )
