"""Tests for DPO HF Jobs training script."""
import sys
from pathlib import Path

# Add scripts/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from train_dpo_hf_job import get_training_script, launch_dpo_training


def test_dpo_script_valid_python():
    script = get_training_script(epochs=1, lr=5e-7, batch_size=4)
    compile(script, "<dpo_job>", "exec")


def test_dpo_script_has_pep723():
    script = get_training_script(epochs=1, lr=5e-7, batch_size=4)
    assert "# /// script" in script


def test_dpo_script_has_trl_dependency():
    script = get_training_script(epochs=1, lr=5e-7, batch_size=4)
    assert "trl" in script


def test_dpo_script_has_peft_dependency():
    script = get_training_script(epochs=1, lr=5e-7, batch_size=4)
    assert "peft" in script


def test_dpo_script_has_lora_config():
    script = get_training_script(epochs=1, lr=5e-7, batch_size=4, lora_rank=32)
    assert "lora_r = 32" in script or "r=32" in script


def test_dpo_script_has_beta():
    script = get_training_script(epochs=1, lr=5e-7, batch_size=4, beta=0.2)
    assert "0.2" in script


def test_dpo_script_data_source_constitutional():
    script = get_training_script(epochs=1, lr=5e-7, batch_size=4, data_source="constitutional")
    assert "constitutional_pairs" in script


def test_dpo_script_data_source_legal():
    script = get_training_script(epochs=1, lr=5e-7, batch_size=4, data_source="legal")
    assert "legal_pairs" in script


def test_launch_function_exists():
    assert callable(launch_dpo_training)
