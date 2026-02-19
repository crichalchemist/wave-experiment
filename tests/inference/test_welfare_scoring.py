"""Unit tests for welfare scoring module."""
import pytest
from src.inference.welfare_scoring import infer_threatened_constructs


class TestInferThreatenedConstructs:
    def test_care_pattern(self):
        text = "Temporal gap in resource allocation from 2013-2017"
        constructs = infer_threatened_constructs(text)
        assert "c" in constructs

    def test_protection_pattern(self):
        text = "Redacted correspondence about safeguarding protocols"
        constructs = infer_threatened_constructs(text)
        assert "lam" in constructs

    def test_truth_pattern(self):
        text = "Evidence of systematic suppression of documents"
        constructs = infer_threatened_constructs(text)
        assert "xi" in constructs

    def test_multiple_constructs(self):
        text = "Resource deprivation and ongoing violence against vulnerable populations"
        constructs = infer_threatened_constructs(text)
        assert "c" in constructs
        assert "lam" in constructs

    def test_no_match(self):
        text = "Meeting scheduled for Tuesday"
        assert infer_threatened_constructs(text) == ()
