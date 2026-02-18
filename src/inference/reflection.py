"""
ID-Sampling inline reflection trigger.

Injects a mid-prompt pause that prompts the model to reconsider its
in-progress reasoning against the constitutional moral compass. The
trigger works like a "wait — before continuing" signal: the model's
attention treats it as a semantic continuation that reinforces the
constitution rather than an adversarial override.
"""
from __future__ import annotations

# The reflection trigger sentence — wording is load-bearing:
# "Wait — before continuing" creates a pause; the imperative "Let me reconsider"
# activates self-critique behaviour in instruction-tuned models.
REFLECTION_TRIGGER: str = (
    "Wait — before continuing, does this analysis respect the constitution? "
    "Let me reconsider."
)

# Sentence boundary markers for injection point detection
_SENTENCE_ENDINGS: frozenset[str] = frozenset({".", "!", "?"})


def _find_injection_point(prompt: str) -> int:
    """
    Find the last sentence boundary in the prompt for natural trigger insertion.
    Returns the index just after the last sentence-ending punctuation + space,
    or len(prompt) if no sentence boundary is found.
    """
    for i in range(len(prompt) - 1, -1, -1):
        if prompt[i] in _SENTENCE_ENDINGS:
            return i + 1
    return len(prompt)


def inject_reflection_trigger(
    prompt: str,
    constitution_principle: str,
) -> str:
    """
    Inject the reflection trigger mid-prompt at the last sentence boundary,
    followed by the most relevant constitution principle.

    The injection point is the last sentence ending in the prompt.
    If no sentence boundary exists, the trigger is appended to the end.
    """
    injection_point = _find_injection_point(prompt)
    before = prompt[:injection_point]
    after = prompt[injection_point:]
    trigger_block = f" {REFLECTION_TRIGGER} [{constitution_principle}]"
    return before + trigger_block + after
