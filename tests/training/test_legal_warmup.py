import json
from unittest.mock import MagicMock, patch


def test_legal_warmup_config_defaults():
    from src.training.legal_warmup import LegalWarmupConfig
    cfg = LegalWarmupConfig()
    assert cfg.examples_per_domain == 200
    assert set(cfg.domains) == {"criminal_justice", "territorial_rights", "foia_transparency"}
    assert cfg.output_path.endswith(".jsonl")


def test_legal_warmup_config_custom_domains():
    from src.training.legal_warmup import LegalWarmupConfig
    cfg = LegalWarmupConfig(domains=("criminal_justice",), examples_per_domain=50)
    assert len(cfg.domains) == 1
    assert cfg.examples_per_domain == 50


def test_run_legal_warmup_calls_pipeline(tmp_path):
    from src.training.legal_warmup import run_legal_warmup, LegalWarmupConfig

    output = str(tmp_path / "legal_pairs.jsonl")
    cfg = LegalWarmupConfig(
        output_path=output,
        examples_per_domain=2,
        domains=("criminal_justice",),
    )

    mock_local = MagicMock()
    mock_local.complete.return_value = "naive analysis"
    mock_critic = MagicMock()
    mock_critic.complete.return_value = "mentor critique"

    fake_docs = [
        {"text": "The court ruled on sentencing guidelines..." * 20, "source": "test", "metadata": {}},
        {"text": "Mandatory minimum sentences diverge from..." * 20, "source": "test", "metadata": {}},
    ]

    with patch("src.training.legal_warmup.load_legal_domain_batch", return_value=fake_docs), \
         patch("src.training.legal_warmup.load_constitution", return_value="constitution text"), \
         patch("src.training.legal_warmup.generate_preference_pair") as mock_gen, \
         patch("src.training.legal_warmup.should_include_example", return_value=True):

        mock_gen.return_value = MagicMock(
            instruction="prompt", rejected="bad", chosen="good"
        )
        count = run_legal_warmup(cfg, mock_local, mock_critic)

    assert count == 2
    lines = (tmp_path / "legal_pairs.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["metadata"]["legal_domain"] == "criminal_justice"
    assert "instruction" in record
    assert "rejected" in record
    assert "chosen" in record


def test_run_legal_warmup_resumes_from_existing(tmp_path):
    from src.training.legal_warmup import run_legal_warmup, LegalWarmupConfig

    output = str(tmp_path / "legal_pairs.jsonl")
    # Pre-populate with 1 existing pair
    (tmp_path / "legal_pairs.jsonl").write_text(
        json.dumps({"instruction": "x", "rejected": "y", "chosen": "z"}) + "\n"
    )

    cfg = LegalWarmupConfig(
        output_path=output,
        examples_per_domain=2,
        domains=("criminal_justice",),
    )

    mock_local = MagicMock()
    mock_local.complete.return_value = "analysis"
    mock_critic = MagicMock()
    mock_critic.complete.return_value = "critique"

    fake_docs = [
        {"text": "Sentencing disparity data shows..." * 20, "source": "test", "metadata": {}},
    ]

    with patch("src.training.legal_warmup.load_legal_domain_batch", return_value=fake_docs), \
         patch("src.training.legal_warmup.load_constitution", return_value="constitution"), \
         patch("src.training.legal_warmup.generate_preference_pair") as mock_gen, \
         patch("src.training.legal_warmup.should_include_example", return_value=True):

        mock_gen.return_value = MagicMock(
            instruction="p", rejected="r", chosen="c"
        )
        count = run_legal_warmup(cfg, mock_local, mock_critic)

    assert count == 1  # only 1 new pair (target was 2, 1 existed)
    lines = (tmp_path / "legal_pairs.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2  # 1 existing + 1 new
