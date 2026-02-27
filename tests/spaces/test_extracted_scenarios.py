"""Test Space integration with extracted scenarios."""
import pytest
import sys
from pathlib import Path


@pytest.fixture(autouse=True)
def add_space_to_path():
    """Add space directory to sys.path for imports."""
    space_dir = str(Path(__file__).parent.parent.parent / "spaces" / "maninagarden")
    if space_dir not in sys.path:
        sys.path.insert(0, space_dir)
    yield
    if space_dir in sys.path:
        sys.path.remove(space_dir)


def test_load_extracted_scenarios_returns_dict():
    from scenarios import load_extracted_scenarios
    result = load_extracted_scenarios()
    assert isinstance(result, dict)
    # No extracted_scenarios.json exists yet → empty dict
    assert result == {}


def test_load_extracted_scenarios_callable():
    from scenarios import load_extracted_scenarios
    assert callable(load_extracted_scenarios)


def test_generate_extracted_scenario_raises_without_file():
    from scenarios import generate_extracted_scenario
    with pytest.raises(ValueError, match="No extracted scenarios"):
        generate_extracted_scenario("nonexistent")


def test_load_with_json_file(tmp_path):
    """When extracted_scenarios.json exists, load_extracted_scenarios reads it."""
    import json
    from unittest.mock import patch
    from scenarios import load_extracted_scenarios, EXTRACTED_SCENARIOS_PATH

    test_data = [
        {"label": "declining_lam_L", "start_levels": {}, "end_levels": {},
         "description": "Love declines over time"}
    ]

    # Write temp JSON and patch the path
    test_file = tmp_path / "extracted_scenarios.json"
    test_file.write_text(json.dumps(test_data))

    with patch("scenarios.EXTRACTED_SCENARIOS_PATH", test_file):
        result = load_extracted_scenarios()

    assert "declining_lam_L" in result
    assert result["declining_lam_L"] == "Love declines over time"
