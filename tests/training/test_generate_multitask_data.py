"""Tests for synthetic multi-task data generator."""
import json


def test_classify_text_detects_assumption_geopolitical():
    from src.training.generate_multitask_data import classify_text
    _, assumption = classify_text("The regulator properly reviewed all disclosures.")
    assert assumption == "geopolitical_presumption"


def test_classify_text_detects_assumption_cognitive():
    from src.training.generate_multitask_data import classify_text
    _, assumption = classify_text("The evidence confirmed our initial assessment.")
    assert assumption == "cognitive_bias"


def test_classify_text_detects_assumption_determinism():
    from src.training.generate_multitask_data import classify_text
    _, assumption = classify_text("Records have always been maintained in chronological order.")
    assert assumption == "historical_determinism"


def test_classify_text_detects_gap_temporal():
    from src.training.generate_multitask_data import classify_text
    gap, _ = classify_text("There is a temporal gap in the records from 2005 to 2008.")
    assert gap == "temporal"


def test_classify_text_detects_gap_evidential():
    from src.training.generate_multitask_data import classify_text
    gap, _ = classify_text("No documentation was found supporting this claim.")
    assert gap == "evidential"


def test_classify_text_returns_none_for_neutral():
    from src.training.generate_multitask_data import classify_text
    gap, assumption = classify_text("The weather was nice today.")
    assert gap is None
    assert assumption is None


def test_generate_synthetic_samples_creates_file(tmp_path):
    from src.training.generate_multitask_data import generate_synthetic_samples
    out = tmp_path / "synthetic.jsonl"
    count = generate_synthetic_samples(count=10, output_path=str(out))
    assert count == 10
    assert out.exists()


def test_generate_synthetic_samples_valid_jsonl(tmp_path):
    from src.training.generate_multitask_data import generate_synthetic_samples
    out = tmp_path / "synthetic.jsonl"
    generate_synthetic_samples(count=5, output_path=str(out))
    lines = [json.loads(line) for line in out.read_text().strip().split("\n")]
    assert len(lines) == 5
    for rec in lines:
        assert "text" in rec
        assert isinstance(rec["text"], str)


def test_generate_synthetic_samples_has_labeled_records(tmp_path):
    from src.training.generate_multitask_data import generate_synthetic_samples
    out = tmp_path / "synthetic.jsonl"
    generate_synthetic_samples(count=50, output_path=str(out))
    lines = [json.loads(line) for line in out.read_text().strip().split("\n")]
    has_gap = any(r.get("gap_type") is not None for r in lines)
    has_assumption = any(r.get("assumption_type") is not None for r in lines)
    assert has_gap, "Expected at least one sample with gap_type"
    assert has_assumption, "Expected at least one sample with assumption_type"
