"""PhiForecaster — dual-head model for Phi + construct prediction.

Uses DignityBackbone (CNN1D + LSTM + Attention) as the shared encoder,
with two ForecastHead decoders:
  - phi_head: predicts scalar Phi trajectory [B, pred_len, 1]
  - construct_head: predicts 8 welfare constructs [B, pred_len, 8]

Multi-task loss: L = L_phi + alpha * L_construct (both MSE).
"""

import sys

if "Dignity-Model" not in sys.path:
    sys.path.insert(0, "Dignity-Model")

import torch
import torch.nn as nn
from models.backbone.hybrid import DignityBackbone
from models.head.forecast import ForecastHead


class PhiForecaster(nn.Module):
    """Dual-head forecaster: Phi scalar + 8 welfare constructs.

    Architecture:
        input [B, T, input_size]
          -> DignityBackbone -> context [B, hidden_size], attn [B, T]
          -> phi_head -> [B, pred_len, 1]
          -> construct_head -> [B, pred_len, 8]

    Args:
        input_size: Number of input features per timestep.
        hidden_size: Hidden dimension for backbone and heads.
        n_layers: Number of LSTM layers in backbone.
        pred_len: Number of future timesteps to predict.
        dropout: Dropout probability.
        alpha: Weight for construct loss in multi-task objective.
    """

    def __init__(
        self,
        input_size: int = 34,
        hidden_size: int = 256,
        n_layers: int = 2,
        pred_len: int = 10,
        dropout: float = 0.1,
        alpha: float = 0.5,
    ):
        super().__init__()

        self.backbone = DignityBackbone(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.phi_head = ForecastHead(
            input_size=hidden_size,
            pred_len=pred_len,
            num_features=1,
            hidden_size=hidden_size // 2,
            dropout=dropout,
        )

        self.construct_head = ForecastHead(
            input_size=hidden_size,
            pred_len=pred_len,
            num_features=8,
            hidden_size=hidden_size // 2,
            dropout=dropout,
        )

        self._loss_fn = nn.MSELoss()
        self.alpha = alpha

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through backbone and both heads.

        Args:
            x: Input tensor [batch_size, seq_len, input_size].

        Returns:
            Tuple of:
              - phi_pred: Phi trajectory [batch_size, pred_len, 1]
              - construct_pred: Construct trajectories [batch_size, pred_len, 8]
              - attn_weights: Attention weights [batch_size, seq_len]
        """
        context, attn_weights = self.backbone(x)
        phi_pred = self.phi_head(context)
        construct_pred = self.construct_head(context)
        return phi_pred, construct_pred, attn_weights

    def predict_phi(self, x: torch.Tensor) -> torch.Tensor:
        """Predict Phi trajectory in eval mode (no gradients).

        Args:
            x: Input tensor [batch_size, seq_len, input_size].

        Returns:
            Phi predictions [batch_size, pred_len, 1], detached.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                phi_pred, _, _ = self.forward(x)
            return phi_pred
        finally:
            if was_training:
                self.train()

    def predict_constructs(self, x: torch.Tensor) -> torch.Tensor:
        """Predict construct trajectories in eval mode (no gradients).

        Args:
            x: Input tensor [batch_size, seq_len, input_size].

        Returns:
            Construct predictions [batch_size, pred_len, 8], detached.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                _, construct_pred, _ = self.forward(x)
            return construct_pred
        finally:
            if was_training:
                self.train()

    def compute_loss(
        self,
        phi_pred: torch.Tensor,
        phi_target: torch.Tensor,
        construct_pred: torch.Tensor,
        construct_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-task loss: L = L_phi + alpha * L_construct.

        Both component losses use MSE.

        Args:
            phi_pred: Predicted Phi [batch_size, pred_len, 1].
            phi_target: Target Phi [batch_size, pred_len, 1].
            construct_pred: Predicted constructs [batch_size, pred_len, 8].
            construct_target: Target constructs [batch_size, pred_len, 8].

        Returns:
            Scalar loss tensor.
        """
        loss_phi = self._loss_fn(phi_pred, phi_target)
        loss_construct = self._loss_fn(construct_pred, construct_target)
        return loss_phi + self.alpha * loss_construct

    @property
    def num_parameters(self) -> int:
        """Total number of trainable + non-trainable parameters."""
        return sum(p.numel() for p in self.parameters())
