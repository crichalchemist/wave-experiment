"""
Model definitions and checkpoint loading for Phi Forecaster.

Model classes are copied VERBATIM from scripts/train_phi_hf_job.py:217-293
to ensure load_state_dict(strict=True) compatibility.
"""
import json
import logging

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_commits

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

MODEL_REPO = "crichalchemist/phi-forecaster"
CHECKPOINT_FILE = "phi_forecaster_best.pt"
SEQ_LEN = 50
PRED_LEN = 10
INPUT_SIZE = 36
HIDDEN_SIZE = 256

# ============================================================================
# Model architecture (verbatim from train_phi_hf_job.py:217-293)
# ============================================================================


class CNN1D(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, num_layers=2, dropout=0.1):
        super().__init__()
        layers = [nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size // 2),
                  nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2),
                       nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        return self.lstm(x)


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        q = self.query(x[:, -1:, :])
        k = self.key(x)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * x).sum(dim=1)
        return context, weights


class PhiForecasterGPU(nn.Module):
    def __init__(self, input_size=36, hidden_size=256, n_layers=2, pred_len=10, dropout=0.1, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.cnn = CNN1D(input_size, hidden_size, kernel_size=3, num_layers=2, dropout=dropout)
        self.lstm = StackedLSTM(hidden_size, hidden_size, n_layers, dropout)
        self.attn = AdditiveAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, pred_len * 1),
        )
        self.construct_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, pred_len * 8),
        )
        self.pred_len = pred_len
        self._loss_fn = nn.MSELoss()

    def forward(self, x):
        h = self.cnn(x)
        h, _ = self.lstm(h)
        h = self.dropout(h)
        context, attn = self.attn(h)
        phi = self.phi_head(context).view(-1, self.pred_len, 1)
        constructs = self.construct_head(context).view(-1, self.pred_len, 8)
        return phi, constructs, attn

    def compute_loss(self, phi_pred, phi_target, construct_pred, construct_target):
        return self._loss_fn(phi_pred, phi_target) + self.alpha * self._loss_fn(construct_pred, construct_target)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Loading functions
# ============================================================================


def load_model(repo_id=MODEL_REPO, filename=CHECKPOINT_FILE, revision=None):
    """Download checkpoint from Hub and load into PhiForecasterGPU."""
    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    model = PhiForecasterGPU(
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
        n_layers=2, pred_len=PRED_LEN,
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def list_checkpoint_versions(repo_id=MODEL_REPO):
    """Return list of dicts with sha, date, message from repo commits (limit 20)."""
    commits = list_repo_commits(repo_id, revision="main")
    results = []
    for commit in commits[:20]:
        results.append({
            "sha": commit.commit_id,
            "date": commit.created_at.isoformat() if commit.created_at else None,
            "message": commit.title,
        })
    return results


def load_training_metadata(repo_id=MODEL_REPO, revision=None):
    """Download and parse training_metadata.json, returns dict (empty dict on failure)."""
    try:
        path = hf_hub_download(repo_id=repo_id, filename="training_metadata.json", revision=revision)
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load training_metadata.json: %s", e)
        return {}
