"""Shared test utilities available to all test modules via pytest auto-discovery."""
from __future__ import annotations

import dataclasses

import pytest


@pytest.fixture()
def assert_frozen():
    """Fixture that returns a helper to verify frozen dataclass immutability.

    Usage::

        def test_my_dataclass_is_frozen(assert_frozen):
            obj = MyFrozenDataclass(field="value")
            assert_frozen(obj, "field", "new_value")
    """
    def _assert_frozen(instance: object, field: str, value: object) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field, value)
    return _assert_frozen
