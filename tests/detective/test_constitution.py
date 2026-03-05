import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.providers import MockProvider
from src.detective.constitution import (
    PreferencePair,
    critique_against_constitution,
    generate_preference_pair,
    load_constitution,
)


def test_load_constitution_reads_file(tmp_path: Path) -> None:
    constitution_file = tmp_path / "constitution.md"
    constitution_file.write_text("# Test Constitution\n\nPrinciple 1.", encoding="utf-8")
    result = load_constitution(path=constitution_file)
    assert "# Test Constitution" in result
    assert "Principle 1." in result


def test_load_constitution_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Constitution not found"):
        load_constitution(path=tmp_path / "nonexistent.md")


def test_critique_against_constitution_calls_provider() -> None:
    critic = MockProvider(response="The analysis collapsed into pattern completion. Here is what stronger reasoning looks like.")
    constitution = "## Core Principles\n\nEpistemic honesty."
    analysis = "The document shows no gaps."
    result = critique_against_constitution(analysis, constitution, critic)
    assert result == "The analysis collapsed into pattern completion. Here is what stronger reasoning looks like."


def test_critique_uses_mentor_prompt() -> None:
    """Verify the critique prompt is built via build_mentor_critique_prompt, not ad-hoc or auditor-style."""
    critic = MockProvider(response="mentor guidance text")
    constitution = "## Constitution"
    analysis = "analysis text"
    with patch("src.detective.constitution.build_mentor_critique_prompt") as mock_build:
        mock_build.return_value = "structured mentor prompt"
        critique_against_constitution(analysis, constitution, critic)
    mock_build.assert_called_once_with(analysis=analysis, constitution=constitution)


def test_mentor_prompt_contains_mentor_role_framing() -> None:
    """Verify the mentor prompt frames Claude as trusted guide, not auditor."""
    from src.security.prompt_guard import build_mentor_critique_prompt
    constitution = "## Core Principles\n\nEpistemic honesty."
    analysis = "The document shows no gaps."
    prompt = build_mentor_critique_prompt(analysis=analysis, constitution=constitution)
    assert "trusted mentor" in prompt
    assert "teach" in prompt
    assert constitution in prompt
    assert analysis in prompt


def test_preference_pair_is_frozen(assert_frozen) -> None:
    pair = PreferencePair(
        instruction="detect gaps",
        chosen="revised analysis",
        rejected="original analysis",
    )
    assert_frozen(pair, "chosen", "modified")


def test_generate_preference_pair_returns_pair() -> None:
    generator = MockProvider(response="Revised: financial records show a gap in Q3 2019.")
    critic = MockProvider(response="The analysis collapsed into pattern completion — the temporal gap in Q3 is the finding.")
    constitution = "## Core Principles\n\nEpistemic honesty above comfort."
    pair = generate_preference_pair(
        instruction="Detect gaps in financial records",
        original_analysis="The financial records appear complete.",
        constitution=constitution,
        generator_provider=generator,
        critic_provider=critic,
    )
    assert isinstance(pair, PreferencePair)
    assert pair.rejected == "The financial records appear complete."
    assert pair.chosen == "Revised: financial records show a gap in Q3 2019."
    assert pair.instruction == "Detect gaps in financial records"


def test_generate_preference_pair_uses_both_providers() -> None:
    # Definitive check: chosen comes from generator, rejected is unchanged original
    pair = generate_preference_pair(
        instruction="test",
        original_analysis="original",
        constitution="## Constitution",
        generator_provider=MockProvider(response="generator_output"),
        critic_provider=MockProvider(response="critic_output"),
    )
    assert pair.chosen == "generator_output"
    assert pair.rejected == "original"


def test_revision_prompt_frames_guidance_as_mentor_direction() -> None:
    """Revision prompt must contain mentor framing, not compliance-checker framing."""
    from unittest.mock import MagicMock
    mock_generator = MagicMock()
    mock_generator.complete.return_value = "revised analysis"
    critic = MockProvider(response="You missed the temporal gap.")
    constitution = "## Core Principles\n\nEpistemic honesty."

    generate_preference_pair(
        instruction="detect gaps",
        original_analysis="no gaps found",
        constitution=constitution,
        generator_provider=mock_generator,
        critic_provider=critic,
    )

    # Inspect the prompt passed to the generator
    revision_prompt = mock_generator.complete.call_args[0][0]
    assert "trusted mentor" in revision_prompt
    assert "You missed the temporal gap." in revision_prompt
