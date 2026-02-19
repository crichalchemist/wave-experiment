"""Tests for constitution-first training pipeline."""
from unittest.mock import MagicMock
from pathlib import Path


def test_import():
    from src.training.constitutional_warmup import (
        run_constitutional_warmup,
        ConstitutionalWarmupConfig,
    )
    assert callable(run_constitutional_warmup)


def test_config_defaults():
    from src.training.constitutional_warmup import ConstitutionalWarmupConfig
    cfg = ConstitutionalWarmupConfig()
    assert cfg.output_path == "data/training/constitutional_pairs.jsonl"
    assert cfg.max_examples == 200


def test_warmup_writes_jsonl(tmp_path, monkeypatch):
    from src.training.constitutional_warmup import run_constitutional_warmup, ConstitutionalWarmupConfig

    # Patch sourcing to return deterministic examples
    mock_examples = [
        {"text": "The regulator properly reviewed all disclosures.", "source": "test", "metadata": {}},
        {"text": "The board always maintained proper records.", "source": "test", "metadata": {}},
    ]
    monkeypatch.setattr(
        "src.training.constitutional_warmup._load_all_sources",
        lambda cfg: mock_examples,
    )

    # Mock welfare filter to always return True for this integration test
    # (Welfare filter behavior is tested separately in TestWelfareFiltering)
    monkeypatch.setattr(
        "src.training.constitutional_warmup.should_include_example",
        lambda text, phi_metrics, welfare_threshold: True,
    )

    local = MagicMock()
    local.complete.return_value = "Analysis: no significant gaps found."
    critic = MagicMock()
    critic.complete.return_value = "Critique: The analysis misses the geopolitical presumption."

    cfg = ConstitutionalWarmupConfig(
        output_path=str(tmp_path / "pairs.jsonl"),
        max_examples=2,
        constitution_path="docs/constitution.md",
    )

    # Patch constitution loading to avoid disk dependency in tests
    monkeypatch.setattr(
        "src.training.constitutional_warmup.load_constitution",
        lambda path=None: "Epistemic honesty above analytical comfort.",
    )

    count = run_constitutional_warmup(cfg, local_provider=local, critic_provider=critic)
    assert count == 2
    output = Path(cfg.output_path)
    assert output.exists()
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 2
    import json
    pair = json.loads(lines[0])
    assert "instruction" in pair
    assert "rejected" in pair
    assert "chosen" in pair


def test_warmup_skips_empty_text(tmp_path, monkeypatch):
    from src.training.constitutional_warmup import run_constitutional_warmup, ConstitutionalWarmupConfig

    monkeypatch.setattr(
        "src.training.constitutional_warmup._load_all_sources",
        lambda cfg: [{"text": "", "source": "test", "metadata": {}}],
    )
    monkeypatch.setattr(
        "src.training.constitutional_warmup.load_constitution",
        lambda path=None: "Constitution text.",
    )

    cfg = ConstitutionalWarmupConfig(output_path=str(tmp_path / "pairs.jsonl"))
    count = run_constitutional_warmup(
        cfg,
        local_provider=MagicMock(),
        critic_provider=MagicMock(),
    )
    assert count == 0


class TestWelfareFiltering:
    def test_includes_welfare_relevant_examples(self):
        """should_include_example returns True for welfare-relevant text."""
        from src.training.constitutional_warmup import should_include_example

        text = "Evidence of resource deprivation affecting vulnerable populations"
        phi_metrics = {"c": 0.3, "lam": 0.3}

        assert should_include_example(text, phi_metrics, welfare_threshold=0.3) is True

    def test_excludes_welfare_irrelevant_examples(self):
        """should_include_example returns False for welfare-irrelevant text."""
        from src.training.constitutional_warmup import should_include_example

        text = "Meeting scheduled for Tuesday at 3pm"
        phi_metrics = {"c": 0.5}

        assert should_include_example(text, phi_metrics, welfare_threshold=0.3) is False

    def test_always_includes_high_urgency_examples(self):
        """should_include_example returns True for high-urgency welfare threats."""
        from src.training.constitutional_warmup import should_include_example

        text = "Evidence of ongoing resource deprivation affecting vulnerable populations"
        phi_metrics = {"c": 0.1, "lam": 0.1}  # very scarce constructs

        # High welfare relevance due to scarce constructs
        # Even with high threshold, scarce constructs boost relevance
        assert should_include_example(text, phi_metrics, welfare_threshold=0.3) is True

    def test_threshold_controls_inclusion(self):
        """welfare_threshold parameter controls inclusion cutoff."""
        from src.training.constitutional_warmup import should_include_example

        text = "Minor evidence of resource issues"
        phi_metrics = {"c": 0.5}

        # Low threshold → includes
        assert should_include_example(text, phi_metrics, welfare_threshold=0.1) is True

        # High threshold → excludes
        assert should_include_example(text, phi_metrics, welfare_threshold=0.9) is False
