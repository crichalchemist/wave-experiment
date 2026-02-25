"""Tests for PhiForecaster — dual-head model for Phi + construct prediction."""

import pytest
import torch

from src.forecasting.model import PhiForecaster


BATCH = 4
SEQ_LEN = 50
INPUT_SIZE = 34
PRED_LEN = 5
HIDDEN_SIZE = 256


@pytest.fixture
def model():
    """Create a PhiForecaster with default test parameters."""
    return PhiForecaster(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        n_layers=2,
        pred_len=PRED_LEN,
        dropout=0.1,
        alpha=0.5,
    )


@pytest.fixture
def sample_input():
    """Create a random input tensor [batch, seq_len, input_size]."""
    return torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)


class TestForwardShapes:
    """Verify output tensor shapes from forward pass."""

    def test_forward_shapes(self, model, sample_input):
        phi_pred, construct_pred, attn_weights = model(sample_input)

        assert phi_pred.shape == (BATCH, PRED_LEN, 1), (
            f"phi_pred shape {phi_pred.shape} != expected ({BATCH}, {PRED_LEN}, 1)"
        )
        assert construct_pred.shape == (BATCH, PRED_LEN, 8), (
            f"construct_pred shape {construct_pred.shape} != expected ({BATCH}, {PRED_LEN}, 8)"
        )
        assert attn_weights.shape == (BATCH, SEQ_LEN), (
            f"attn_weights shape {attn_weights.shape} != expected ({BATCH}, {SEQ_LEN})"
        )


class TestPredictPhi:
    """Verify predict_phi convenience method."""

    def test_predict_phi(self, model):
        x = torch.randn(2, SEQ_LEN, INPUT_SIZE)
        phi = model.predict_phi(x)

        assert phi.shape == (2, PRED_LEN, 1)
        assert not phi.requires_grad, "predict_phi should return detached tensor"

    def test_predict_phi_eval_mode(self, model):
        """predict_phi should temporarily set model to eval mode."""
        model.train()
        x = torch.randn(2, SEQ_LEN, INPUT_SIZE)
        _ = model.predict_phi(x)
        # Model should be restored to its original training state
        assert model.training, "predict_phi should restore original training state"


class TestPredictConstructs:
    """Verify predict_constructs convenience method."""

    def test_predict_constructs(self, model):
        x = torch.randn(2, SEQ_LEN, INPUT_SIZE)
        constructs = model.predict_constructs(x)

        assert constructs.shape == (2, PRED_LEN, 8)
        assert not constructs.requires_grad, "predict_constructs should return detached tensor"

    def test_predict_constructs_eval_mode(self, model):
        """predict_constructs should temporarily set model to eval mode."""
        model.train()
        x = torch.randn(2, SEQ_LEN, INPUT_SIZE)
        _ = model.predict_constructs(x)
        assert model.training, "predict_constructs should restore original training state"


class TestGradientFlow:
    """Verify gradients flow through entire model."""

    def test_gradient_flows(self, model, sample_input):
        phi_pred, construct_pred, _ = model(sample_input)

        phi_target = torch.randn_like(phi_pred)
        construct_target = torch.randn_like(construct_pred)

        loss = model.compute_loss(phi_pred, phi_target, construct_pred, construct_target)
        loss.backward()

        # Check that backbone parameters received gradients
        backbone_has_grad = False
        for param in model.backbone.parameters():
            if param.grad is not None:
                backbone_has_grad = True
                break

        assert backbone_has_grad, "Backbone parameters should have non-None gradients after backward"

        # Check both heads received gradients
        phi_head_has_grad = any(p.grad is not None for p in model.phi_head.parameters())
        construct_head_has_grad = any(p.grad is not None for p in model.construct_head.parameters())

        assert phi_head_has_grad, "Phi head should have gradients"
        assert construct_head_has_grad, "Construct head should have gradients"


class TestMultiTaskLoss:
    """Verify multi-task loss computation."""

    def test_multi_task_loss(self, model, sample_input):
        phi_pred, construct_pred, _ = model(sample_input)

        phi_target = torch.randn_like(phi_pred)
        construct_target = torch.randn_like(construct_pred)

        loss = model.compute_loss(phi_pred, phi_target, construct_pred, construct_target)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert loss.requires_grad, "Loss should be differentiable"

    def test_loss_alpha_weighting(self):
        """Verify that alpha controls the construct loss weight."""
        model_low = PhiForecaster(input_size=INPUT_SIZE, pred_len=PRED_LEN, alpha=0.0)
        model_high = PhiForecaster(input_size=INPUT_SIZE, pred_len=PRED_LEN, alpha=1.0)

        x = torch.randn(2, SEQ_LEN, INPUT_SIZE)

        # Use same input for both models (different weights, but we test the formula)
        phi_pred = torch.randn(2, PRED_LEN, 1)
        construct_pred = torch.randn(2, PRED_LEN, 8)

        phi_target = torch.zeros(2, PRED_LEN, 1)
        construct_target = torch.zeros(2, PRED_LEN, 8)

        loss_zero_alpha = model_low.compute_loss(phi_pred, phi_target, construct_pred, construct_target)
        loss_one_alpha = model_high.compute_loss(phi_pred, phi_target, construct_pred, construct_target)

        # With alpha=0, construct loss is ignored; with alpha=1, it's fully weighted
        # Both losses use the same predictions, so we can compare
        assert loss_one_alpha.item() >= loss_zero_alpha.item(), (
            "Higher alpha should produce equal or larger total loss"
        )


class TestNumParameters:
    """Verify num_parameters property."""

    def test_num_parameters(self, model):
        n = model.num_parameters
        assert isinstance(n, int)
        assert n > 0, "Model should have parameters"

    def test_num_parameters_matches_pytorch(self, model):
        expected = sum(p.numel() for p in model.parameters())
        assert model.num_parameters == expected
