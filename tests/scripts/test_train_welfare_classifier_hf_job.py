"""Tests for the HF Jobs welfare classifier training script generator."""

import pytest


def test_script_generation_returns_valid_python():
    from scripts.train_welfare_classifier_hf_job import get_training_script

    script = get_training_script(epochs=1, lr=2e-5, batch_size=16)
    assert "distilbert-base-uncased" in script
    assert "welfare-constructs-distilbert" in script
    compile(script, "<hf_job>", "exec")


def test_script_includes_pep723_dependencies():
    from scripts.train_welfare_classifier_hf_job import get_training_script

    script = get_training_script(epochs=1, lr=2e-5, batch_size=16)
    assert "# /// script" in script
    assert "torch" in script
    assert "transformers" in script


def test_script_includes_data_download():
    from scripts.train_welfare_classifier_hf_job import get_training_script

    script = get_training_script(epochs=1, lr=2e-5, batch_size=16)
    assert "huggingface_hub" in script or "hf_hub_download" in script


def test_script_pushes_to_hub():
    from scripts.train_welfare_classifier_hf_job import get_training_script

    script = get_training_script(epochs=1, lr=2e-5, batch_size=16)
    assert "push_to_hub" in script or "upload" in script


def test_launch_function_exists():
    from scripts.train_welfare_classifier_hf_job import launch_classifier_training

    assert callable(launch_classifier_training)
