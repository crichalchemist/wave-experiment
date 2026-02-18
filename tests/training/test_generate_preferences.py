"""Tests for constitutional preference pair generator."""
from __future__ import annotations

import json
import pytest

from src.core.providers import MockProvider


def test_generate_analysis_calls_provider():
    from src.training.generate_preferences import generate_analysis

    provider = MockProvider(response="Gap detected in 2013 records.")
    result = generate_analysis("Find gaps", "Entity A was active.", provider)
    assert "Gap detected" in result


def test_generate_analysis_includes_text_in_prompt():
    """Verify the document text appears in the prompt sent to the provider."""
    from src.training.generate_preferences import generate_analysis

    calls: list[str] = []

    class CapturingProvider:
        def complete(self, prompt: str, **_) -> str:
            calls.append(prompt)
            return "ok"

        def embed(self, text: str) -> list[float]:
            return []

    generate_analysis("Find gaps", "UNIQUE_DOCUMENT_TEXT_MARKER", CapturingProvider())
    assert "UNIQUE_DOCUMENT_TEXT_MARKER" in calls[0]


def test_preference_pair_format(tmp_path):
    """Each JSONL line has instruction, rejected, chosen."""
    from src.training.generate_preferences import (
        AnnotationExample,
        generate_preferences_to_jsonl,
    )

    examples = [
        AnnotationExample("Find gaps", "Entity A was active in 2013.", "temporal"),
    ]
    local = MockProvider(response="Analysis: no gaps found.")
    critic = MockProvider(response="Revised: a temporal gap exists in 2013.")
    output_file = tmp_path / "prefs.jsonl"

    count = generate_preferences_to_jsonl(
        examples,
        local,
        critic,
        output_path=str(output_file),
        constitution="Principle: Be honest.",
    )

    assert count == 1
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 1
    pair = json.loads(lines[0])
    assert "instruction" in pair
    assert "rejected" in pair
    assert "chosen" in pair


def test_multiple_examples_write_multiple_lines(tmp_path):
    from src.training.generate_preferences import (
        AnnotationExample,
        generate_preferences_to_jsonl,
    )

    examples = [
        AnnotationExample(f"Instruction {i}", f"Text {i}", "temporal")
        for i in range(3)
    ]
    local = MockProvider(response="analysis")
    critic = MockProvider(response="revised analysis")
    output_file = tmp_path / "prefs.jsonl"

    count = generate_preferences_to_jsonl(
        examples,
        local,
        critic,
        output_path=str(output_file),
        constitution="Principle: X.",
    )

    assert count == 3
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        pair = json.loads(line)
        assert set(pair.keys()) == {"instruction", "rejected", "chosen"}


def test_annotation_example_is_immutable():
    from src.training.generate_preferences import AnnotationExample

    ex = AnnotationExample("instr", "text", "temporal")
    with pytest.raises((AttributeError, TypeError)):
        ex.instruction = "other"  # type: ignore[misc]


def test_output_directory_created_if_missing(tmp_path):
    """generate_preferences_to_jsonl creates the output directory."""
    from src.training.generate_preferences import (
        AnnotationExample,
        generate_preferences_to_jsonl,
    )

    deep_path = tmp_path / "a" / "b" / "c" / "prefs.jsonl"
    count = generate_preferences_to_jsonl(
        [AnnotationExample("i", "t", "temporal")],
        MockProvider(response="r"),
        MockProvider(response="r2"),
        output_path=str(deep_path),
        constitution="P: X.",
    )
    assert count == 1
    assert deep_path.exists()
