"""Forecast backbone — CNN1D + StackedLSTM + AdditiveAttention encoder.

Extracted from Dignity-Model/models/ to remove the embedded repository
dependency.  Contains five PyTorch modules:

- CNN1D: 1D convolutions for local temporal pattern extraction
- StackedLSTM: multi-layer LSTM for long-term temporal dependencies
- AdditiveAttention: Bahdanau-style attention for sequence weighting
- ForecastBackbone: composable CNN → LSTM → Attention encoder
- ForecastHead: feedforward decoder for multi-step predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class CNN1D(nn.Module):
    """
    1D Convolutional Network for extracting local temporal patterns.

    Processes sequences with multiple convolutional layers to capture
    short-term dependencies and local features.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of CNN filters
            kernel_size: Convolution kernel size
            num_layers: Number of CNN layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        # Build CNN layers
        layers = []

        # First layer: input_size -> hidden_size
        layers.append(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Additional layers: hidden_size -> hidden_size
        for _ in range(num_layers - 1):
            layers.append(
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.cnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN layers.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Conv1d expects [B, C, L] format
        x = x.transpose(1, 2)  # [B, input_size, seq_len]

        # Apply convolutions
        x = self.cnn(x)  # [B, hidden_size, seq_len]

        # Transpose back to [B, seq_len, hidden_size]
        x = x.transpose(1, 2)

        return x


class StackedLSTM(nn.Module):
    """
    Stacked LSTM for capturing long-term temporal dependencies.

    Processes sequences through multiple LSTM layers to learn
    complex temporal patterns and dependencies.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability between layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output size adjustment for bidirectional
        self.output_size = hidden_size * 2 if bidirectional else hidden_size

    def forward(self, x: torch.Tensor, hidden: tuple = None) -> tuple:
        """
        Forward pass through LSTM layers.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            hidden: Optional initial hidden state tuple (h_0, c_0)

        Returns:
            Tuple of (output, (h_n, c_n))
            - output: [batch_size, seq_len, hidden_size * num_directions]
            - h_n: [num_layers * num_directions, batch_size, hidden_size]
            - c_n: [num_layers * num_directions, batch_size, hidden_size]
        """
        output, (h_n, c_n) = self.lstm(x, hidden)
        return output, (h_n, c_n)

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (h_0, c_0) initialized to zeros
        """
        num_directions = 2 if self.bidirectional else 1
        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        return (h_0, c_0)


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention mechanism.

    Computes attention weights over sequence to produce
    a weighted context vector focusing on important timesteps.
    """

    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size: Size of hidden representations
        """
        super().__init__()

        self.hidden_size = hidden_size

        # Attention parameters
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Compute attention-weighted context vector.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            mask: Optional mask [batch_size, seq_len] (1 = keep, 0 = mask)

        Returns:
            Tuple of:
            - context: Attention-weighted sum [batch_size, hidden_size]
            - weights: Attention weights [batch_size, seq_len]
        """
        # Compute attention scores
        # [B, T, H] -> [B, T, H] -> [B, T, 1]
        scores = self.v(torch.tanh(self.W(hidden_states)))
        scores = scores.squeeze(-1)  # [B, T]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        weights = F.softmax(scores, dim=1)  # [B, T]

        # Compute weighted context
        # [B, T] -> [B, T, 1] * [B, T, H] -> [B, T, H] -> [B, H]
        context = torch.sum(weights.unsqueeze(-1) * hidden_states, dim=1)

        return context, weights


# ---------------------------------------------------------------------------
# Backbone (encoder)
# ---------------------------------------------------------------------------


class ForecastBackbone(nn.Module):
    """
    Composable backbone combining CNN, LSTM, and Attention.

    Architecture:
    1. 1D-CNN: Extract local temporal patterns
    2. Stacked LSTM: Model long-term dependencies
    3. Additive Attention: Focus on important timesteps
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        cnn_kernel_size: int = 3,
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden size for CNN/LSTM
            n_layers: Number of LSTM layers
            dropout: Dropout probability
            cnn_kernel_size: CNN kernel size
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_checkpointing = False  # Set by config, trades compute for memory

        # 1. CNN for local pattern extraction
        self.cnn = CNN1D(
            input_size=input_size,
            hidden_size=hidden_size,
            kernel_size=cnn_kernel_size,
            num_layers=2,
            dropout=dropout,
        )

        # 2. LSTM for temporal modeling
        self.lstm = StackedLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=False,
        )

        # 3. Attention for sequence weighting
        self.attn = AdditiveAttention(hidden_size)

        # 4. Dropout regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Forward pass through the backbone.

        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            mask: Optional mask [batch_size, seq_len]

        Returns:
            Tuple of:
            - context: Context vector [batch_size, hidden_size]
            - attention_weights: [batch_size, seq_len]
        """
        # 1. CNN: Extract local features (checkpoint if enabled and training)
        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.cnn, x, use_reentrant=False
            )  # [B, T, H]
        else:
            x = self.cnn(x)  # [B, T, H]

        # 2. LSTM: Model temporal dependencies (checkpoint if enabled and training)
        if self.use_checkpointing and self.training:
            # LSTM returns (output, hidden), we only need output
            x = torch.utils.checkpoint.checkpoint(
                lambda x: self.lstm(x)[0], x, use_reentrant=False
            )  # [B, T, H]
        else:
            x, _ = self.lstm(x)  # [B, T, H]

        x = self.dropout(x)

        # 3. Attention: Compute weighted context (don't checkpoint - needs gradients)
        context, attn_weights = self.attn(x, mask)  # [B, H], [B, T]

        return context, attn_weights


# ---------------------------------------------------------------------------
# Head (decoder)
# ---------------------------------------------------------------------------


class ForecastHead(nn.Module):
    """
    Task head for forecasting future values.

    Predicts multiple timesteps ahead from a context vector
    via a feedforward network.
    """

    def __init__(
        self,
        input_size: int,
        pred_len: int = 5,
        num_features: int = 3,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_size: Size of backbone context vector
            pred_len: Number of timesteps to predict
            num_features: Number of features to forecast per timestep
            hidden_size: Size of intermediate layer
            dropout: Dropout probability
        """
        super().__init__()

        self.pred_len = pred_len
        self.num_features = num_features

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len * num_features),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Forecast future values from context vector.

        Args:
            context: Context vector [batch_size, input_size]

        Returns:
            Forecasts [batch_size, pred_len, num_features]
        """
        batch_size = context.size(0)

        # Generate flat predictions
        pred_flat = self.net(context)  # [B, pred_len * num_features]

        # Reshape to [B, pred_len, num_features]
        predictions = pred_flat.view(batch_size, self.pred_len, self.num_features)

        return predictions
