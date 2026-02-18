"""Tests for inline reflection trigger injection."""


def test_reflection_trigger_constant_contains_key_phrases():
    from src.inference.reflection import REFLECTION_TRIGGER
    assert "Wait" in REFLECTION_TRIGGER
    assert "reconsider" in REFLECTION_TRIGGER
    assert "constitution" in REFLECTION_TRIGGER


def test_inject_adds_trigger_to_prompt():
    from src.inference.reflection import inject_reflection_trigger
    result = inject_reflection_trigger(
        "Entity A was active in 2013.",
        "All claims must cite evidence."
    )
    assert "Wait" in result
    assert "reconsider" in result


def test_inject_includes_constitution_principle():
    from src.inference.reflection import inject_reflection_trigger
    result = inject_reflection_trigger(
        "Some analysis here.",
        "PRINCIPLE_XYZ"
    )
    assert "PRINCIPLE_XYZ" in result


def test_inject_preserves_original_content():
    from src.inference.reflection import inject_reflection_trigger
    original = "Entity A influenced Policy X."
    result = inject_reflection_trigger(original, "Be honest.")
    assert "Entity A influenced Policy X" in result


def test_inject_at_sentence_boundary():
    """Trigger should be inserted at a sentence boundary, not appended to end."""
    from src.inference.reflection import inject_reflection_trigger
    prompt = "First sentence. Second sentence."
    result = inject_reflection_trigger(prompt, "principle")
    # The trigger should appear somewhere before the very end
    trigger_pos = result.find("Wait")
    assert trigger_pos < len(result) - 5  # not at the very end


def test_inject_no_sentence_boundary_appends():
    """If no sentence boundary exists, trigger is appended."""
    from src.inference.reflection import inject_reflection_trigger
    prompt = "no punctuation here"
    result = inject_reflection_trigger(prompt, "principle")
    assert result.startswith(prompt)
    assert "Wait" in result


def test_find_injection_point_returns_after_last_period():
    from src.inference.reflection import _find_injection_point
    prompt = "First. Second. Third."
    point = _find_injection_point(prompt)
    # Should point just after the last period
    assert prompt[point - 1] == "."


def test_find_injection_point_no_boundary_returns_length():
    from src.inference.reflection import _find_injection_point
    prompt = "no sentence ending here"
    assert _find_injection_point(prompt) == len(prompt)
