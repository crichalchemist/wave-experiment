"""Tests for hypothesis evolution."""

import pytest
from src.detective.hypothesis import Hypothesis


def test_hypothesis_creation():
    """Test creating a hypothesis."""
    h = Hypothesis.create("Test hypothesis", 0.8)
    assert h.confidence == 0.8
    assert h.text == "Test hypothesis"
    assert h.parent_id is None


def test_hypothesis_update():
    """Test updating hypothesis confidence."""
    h1 = Hypothesis.create("Test", 0.8)
    h2 = h1.update_confidence(0.6)
    
    assert h2.confidence == 0.6
    assert h2.parent_id == h1.id
    assert h1.confidence == 0.8  # Original unchanged
