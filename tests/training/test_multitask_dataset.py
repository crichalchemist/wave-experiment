"""Tests for multi-task dataset loading."""
import json

import pytest


def test_gap_type_to_index_has_five_entries():
    from src.training.multitask_dataset import GAP_TYPE_TO_INDEX
    assert len(GAP_TYPE_TO_INDEX) == 5


def test_assumption_type_to_index_has_three_entries():
    from src.training.multitask_dataset import ASSUMPTION_TYPE_TO_INDEX
    assert len(ASSUMPTION_TYPE_TO_INDEX) == 3


def test_gap_type_to_index_covers_all_enum_values():
    from src.training.multitask_dataset import GAP_TYPE_TO_INDEX
    from src.core.types import GapType
    for gt in GapType:
        assert gt.value in GAP_TYPE_TO_INDEX, f"Missing GapType: {gt.value}"


def test_assumption_type_to_index_covers_all_enum_values():
    from src.training.multitask_dataset import ASSUMPTION_TYPE_TO_INDEX
    from src.core.types import AssumptionType
    for at in AssumptionType:
        assert at.value in ASSUMPTION_TYPE_TO_INDEX, f"Missing AssumptionType: {at.value}"


def test_multitask_sample_is_frozen():
    from src.training.multitask_dataset import MultitaskSample
    s = MultitaskSample(text="hello")
    with pytest.raises((AttributeError, TypeError)):
        s.text = "other"  # type: ignore[misc]


def test_multitask_sample_defaults():
    from src.training.multitask_dataset import MultitaskSample
    s = MultitaskSample(text="hello")
    assert s.text == "hello"
    assert s.gap_type is None
    assert s.assumption_type is None


def test_load_multitask_annotations_valid(tmp_path):
    from src.training.multitask_dataset import load_multitask_annotations
    data = [
        {"text": "Entity was active.", "gap_type": "temporal"},
        {"text": "The regulator properly reviewed.", "assumption_type": "geopolitical_presumption"},
    ]
    f = tmp_path / "annot.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in data))
    result = load_multitask_annotations(str(f))
    assert len(result) == 2
    assert result[0].gap_type == "temporal"
    assert result[0].assumption_type is None
    assert result[1].assumption_type == "geopolitical_presumption"
    assert result[1].gap_type is None


def test_load_multitask_annotations_missing_file():
    from src.training.multitask_dataset import load_multitask_annotations
    with pytest.raises(FileNotFoundError):
        load_multitask_annotations("/nonexistent/path.jsonl")


def test_load_multitask_annotations_skips_blank_lines(tmp_path):
    from src.training.multitask_dataset import load_multitask_annotations
    f = tmp_path / "annot.jsonl"
    f.write_text('{"text": "hello"}\n\n{"text": "world"}\n')
    result = load_multitask_annotations(str(f))
    assert len(result) == 2


def test_tokenize_sample_basic():
    from src.training.multitask_dataset import tokenize_sample
    char_to_id = {"a": 0, "b": 1, "c": 2}
    tokens = tokenize_sample("abc", char_to_id, bos_id=3, max_len=10)
    assert tokens[0] == 3  # BOS
    assert tokens[-1] == 3  # BOS
    assert len(tokens) == 5  # BOS + a + b + c + BOS


def test_tokenize_sample_truncates():
    from src.training.multitask_dataset import tokenize_sample
    char_to_id = {chr(i): i for i in range(128)}
    tokens = tokenize_sample("abcdefgh", char_to_id, bos_id=200, max_len=5)
    assert len(tokens) == 5


def test_tokenize_sample_unknown_chars():
    from src.training.multitask_dataset import tokenize_sample
    char_to_id = {"a": 0}
    tokens = tokenize_sample("abc", char_to_id, bos_id=99, max_len=10)
    # 'b' and 'c' are unknown, should map to bos_id
    assert tokens == [99, 0, 99, 99, 99]
