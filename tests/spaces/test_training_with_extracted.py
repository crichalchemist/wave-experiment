"""Test training script includes extracted scenarios."""
import importlib.util
import sys
from pathlib import Path
import pytest


SPACE_DIR = Path(__file__).parent.parent.parent / "spaces" / "maninagarden"


@pytest.fixture()
def training_module():
    """Import training.py from the Space directory without polluting sys.path."""
    # Import welfare first (training.py depends on it)
    welfare_path = SPACE_DIR / "welfare.py"
    welfare_spec = importlib.util.spec_from_file_location("welfare", welfare_path)
    welfare_mod = importlib.util.module_from_spec(welfare_spec)
    sys.modules["welfare"] = welfare_mod
    welfare_spec.loader.exec_module(welfare_mod)

    training_path = SPACE_DIR / "training.py"
    spec = importlib.util.spec_from_file_location("space_training", training_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod

    # Cleanup
    sys.modules.pop("welfare", None)


def test_training_script_references_extracted_scenarios(training_module):
    script = training_module.get_training_script(
        epochs=1, lr=1e-3, hidden_size=128,
        batch_size=32, scenarios_per_type=10,
    )
    assert "extracted_scenarios" in script or "extracted_templates" in script


def test_training_script_valid_python(training_module):
    script = training_module.get_training_script(
        epochs=1, lr=1e-3, hidden_size=128,
        batch_size=32, scenarios_per_type=10,
    )
    compile(script, "<training_script>", "exec")  # valid Python
