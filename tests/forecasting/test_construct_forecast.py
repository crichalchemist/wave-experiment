"""Tests for construct forecast helper."""
from src.forecasting.construct_forecast import forecasted_metrics_to_dict
from src.inference.welfare_scoring import ALL_CONSTRUCTS


def test_all_constructs_mapped():
    """Call forecasted_metrics_to_dict with [0.1]*8, verify all 8 construct keys present."""
    result = forecasted_metrics_to_dict([0.1] * 8)
    assert len(result) == 8
    for construct in ALL_CONSTRUCTS:
        assert construct in result


def test_correct_positional_mapping():
    """Create array with distinct values, verify each construct maps to the correct value."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    result = forecasted_metrics_to_dict(values)

    for i, construct in enumerate(ALL_CONSTRUCTS):
        assert result[construct] == values[i], (
            f"Construct {construct} at index {i} should be {values[i]}, got {result[construct]}"
        )


def test_float_conversion():
    """Pass integers [1, 2, 3, 4, 5, 6, 7, 8], verify all values in result are floats."""
    result = forecasted_metrics_to_dict([1, 2, 3, 4, 5, 6, 7, 8])
    for construct, value in result.items():
        assert isinstance(value, float), (
            f"Value for {construct} should be float, got {type(value).__name__}"
        )
