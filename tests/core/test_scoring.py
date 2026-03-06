import pytest
from src.core.scoring import parse_score, clamp_confidence, SCORE_RE


class TestParseScore:
    def test_standard_format(self):
        assert parse_score("score: 0.85") == 0.85

    def test_confidence_format(self):
        assert parse_score("confidence: 0.7") == 0.7

    def test_equals_separator(self):
        assert parse_score("score = 0.6") == 0.6

    def test_clamps_above_one(self):
        assert parse_score("score: 1.5") == 1.0

    def test_clamps_below_zero(self):
        assert parse_score("score: -0.3") == 0.0

    def test_no_match_returns_default(self):
        assert parse_score("no score here") == 0.0

    def test_custom_default(self):
        assert parse_score("no score here", default=0.5) == 0.5

    def test_integer_score(self):
        assert parse_score("score: 1") == 1.0

    def test_case_insensitive(self):
        assert parse_score("Score: 0.9") == 0.9
        assert parse_score("SCORE: 0.9") == 0.9


class TestClampConfidence:
    def test_within_range(self):
        assert clamp_confidence(0.5) == 0.5

    def test_above_one(self):
        assert clamp_confidence(1.5) == 1.0

    def test_below_zero(self):
        assert clamp_confidence(-0.5) == 0.0

    def test_boundary_zero(self):
        assert clamp_confidence(0.0) == 0.0

    def test_boundary_one(self):
        assert clamp_confidence(1.0) == 1.0
