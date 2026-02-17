from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from src.core.providers import ModelProvider
from src.security.prompt_guard import build_mentor_critique_prompt, build_revision_prompt

_DEFAULT_CONSTITUTION_PATH: Path = Path("docs/constitution.md")


def load_constitution(path: Path = _DEFAULT_CONSTITUTION_PATH) -> str:
    """
    Load constitution from disk. Constitution is the epistemic foundation —
    loading it fresh ensures the critique loop reflects any revisions.
    """
    if not path.exists():
        raise FileNotFoundError(f"Constitution not found at {path}. This is required for CAI.")
    return path.read_text(encoding="utf-8")


def critique_against_constitution(
    analysis: str,
    constitution: str,
    critic_provider: ModelProvider,
) -> str:
    """
    Critique an analysis using Claude as trusted mentor, not mere auditor.
    Claude's internalized values produce directional guidance the local model can
    learn from — stronger training signal than a checklist-style pass/fail critique.
    """
    prompt = build_mentor_critique_prompt(analysis=analysis, constitution=constitution)
    return critic_provider.complete(prompt)


@dataclass(frozen=True)
class PreferencePair:
    """
    A training example for constitutional preference learning.
    chosen is the revised (epistemically honest) analysis;
    rejected is the original that failed the critique.
    """
    instruction: str
    chosen: str     # revised analysis — learned from mentor guidance
    rejected: str   # original analysis — failed the critique


def generate_preference_pair(
    instruction: str,
    original_analysis: str,
    constitution: str,
    generator_provider: ModelProvider,
    critic_provider: ModelProvider,
) -> PreferencePair:
    """
    Full CAI loop: generate → mentor critique → revise with internalized guidance → preference pair.
    Separation of generator (local vLLM) and mentor (external Claude) is load-bearing —
    Claude's externally-trained values produce a stronger critique signal than same-model
    self-critique, and the mentor framing shapes how the generator incorporates that signal.
    """
    critique = critique_against_constitution(original_analysis, constitution, critic_provider)

    revision_prompt = build_revision_prompt(
        original_analysis=original_analysis,
        critique=critique,
        constitution=constitution,
    )
    revised_analysis = generator_provider.complete(revision_prompt)

    return PreferencePair(
        instruction=instruction,
        chosen=revised_analysis,
        rejected=original_analysis,
    )
