# /// script
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "pandas>=2.0.0",
#     "scikit-learn>=1.3.0",
#     "scipy>=1.10.0",
#     "tqdm>=4.65.0",
#     "huggingface-hub>=0.25.0",
#     "trackio",
# ]
# ///
"""Train PhiForecaster on synthetic Phi trajectories — HF Jobs GPU version.

Generates 8 welfare scenarios x 50 instances, trains CNN+LSTM+Attention
dual-head model for 100 epochs, logs metrics with Trackio, saves to Hub.
"""
import math
import os
import json
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Inline welfare scoring (self-contained — no external imports on HF Jobs)
# ============================================================================

ALL_CONSTRUCTS = ("c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi")

CONSTRUCT_FLOORS = {
    "c": 0.20, "kappa": 0.20, "lam_P": 0.20, "lam_L": 0.15,
    "xi": 0.30, "j": 0.10, "p": 0.10, "eps": 0.10,
}

ETA = 0.10
MU = 0.15
ETA_CURIOSITY = 0.08
GAMMA = 0.5

CONSTRUCT_PAIRS = [("c", "lam_L"), ("kappa", "lam_P"), ("j", "p"), ("eps", "xi")]
CURIOSITY_CROSS_PAIR = ("lam_L", "xi")
PENALTY_PAIRS = CONSTRUCT_PAIRS + [CURIOSITY_CROSS_PAIR]


def compute_phi(metrics: Dict[str, float]) -> float:
    lam_L = max(0.01, metrics.get("lam_L", 0.5))
    f_lam = max(0.01, lam_L) ** GAMMA
    inv = {c: 1.0 / max(0.01, metrics.get(c, 0.5)) for c in ALL_CONSTRUCTS}
    inv_sum = sum(inv.values())
    weights = {c: inv[c] / inv_sum for c in ALL_CONSTRUCTS}
    product = 1.0
    for c in ALL_CONSTRUCTS:
        x = max(0.01, metrics.get(c, 0.5))
        product *= x ** weights[c]
    pair_sum = sum(
        math.sqrt(max(0.0, metrics.get(a, 0.5)) * max(0.0, metrics.get(b, 0.5)))
        for a, b in CONSTRUCT_PAIRS
    )
    a, b = CURIOSITY_CROSS_PAIR
    curiosity = math.sqrt(max(0.0, metrics.get(a, 0.5)) * max(0.0, metrics.get(b, 0.5)))
    synergy = 1.0 + ETA * pair_sum + ETA_CURIOSITY * curiosity
    sq_sum = sum((metrics.get(a, 0.5) - metrics.get(b, 0.5)) ** 2 for a, b in PENALTY_PAIRS)
    penalty = MU * sq_sum / len(PENALTY_PAIRS)
    return max(0.0, f_lam * product * synergy * (1.0 - penalty))


# ============================================================================
# Inline signal processing
# ============================================================================

def volatility(values, window=20):
    if len(values) < window:
        return np.zeros_like(values)
    result = np.zeros_like(values, dtype=np.float64)
    for i in range(window, len(values)):
        result[i] = np.std(values[i - window:i])
    if window < len(values):
        result[:window] = result[window]
    return result


def price_momentum(prices, window=10):
    if len(prices) < window:
        return np.zeros_like(prices)
    result = np.zeros_like(prices, dtype=np.float64)
    for i in range(window, len(prices)):
        result[i] = (prices[i] - prices[i - window]) / max(1e-10, abs(prices[i - window]))
    return result


def synergy_signal(a, b, window=20):
    geo = np.sqrt(np.maximum(0, a) * np.maximum(0, b))
    if len(geo) < window:
        return geo
    result = np.zeros_like(geo, dtype=np.float64)
    for i in range(window, len(geo)):
        result[i] = np.mean(geo[i - window:i])
    result[:window] = result[window] if window < len(geo) else 0
    return result


def divergence_signal(a, b, window=20):
    sq_diff = (a - b) ** 2
    if len(sq_diff) < window:
        return sq_diff
    result = np.zeros_like(sq_diff, dtype=np.float64)
    for i in range(window, len(sq_diff)):
        result[i] = np.mean(sq_diff[i - window:i])
    result[:window] = result[window] if window < len(sq_diff) else 0
    return result


def compute_all_signals(df, window=20):
    out = df[list(ALL_CONSTRUCTS)].copy()
    for c in ALL_CONSTRUCTS:
        vals = out[c].values.astype(np.float64)
        out[f"{c}_vol"] = volatility(vals, window)
        out[f"{c}_mom"] = price_momentum(vals, window)
    for a, b in PENALTY_PAIRS:
        out[f"syn_{a}_{b}"] = synergy_signal(df[a].values, df[b].values, window)
        out[f"div_{a}_{b}"] = divergence_signal(df[a].values, df[b].values, window)
    phi_vals = np.array([compute_phi({c: row[c] for c in ALL_CONSTRUCTS}) for _, row in out[list(ALL_CONSTRUCTS)].iterrows()])
    out["phi"] = phi_vals
    out["dphi_dt"] = np.gradient(phi_vals)
    return out


# ============================================================================
# Inline synthetic generator
# ============================================================================

SCENARIOS = (
    "stable_community", "capitalism_suppresses_love", "surveillance_state",
    "willful_ignorance", "recovery_arc", "sudden_crisis", "slow_decay", "random_walk",
)


def generate_scenario(scenario, length=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    def base(noise=0.01):
        return {c: 0.5 + rng.normal(0, noise, length).cumsum().clip(-0.1, 0.1) for c in ALL_CONSTRUCTS}

    if scenario == "stable_community":
        d = {c: 0.5 + rng.normal(0, 0.02, length).cumsum().clip(-0.1, 0.1) for c in ALL_CONSTRUCTS}
    elif scenario == "capitalism_suppresses_love":
        d = base()
        d["lam_L"] = np.linspace(0.6, 0.1, length) + rng.normal(0, 0.01, length)
        d["p"] = np.linspace(0.5, 0.25, length) + rng.normal(0, 0.01, length)
    elif scenario == "surveillance_state":
        d = base()
        d["xi"] = np.linspace(0.3, 0.9, length) + rng.normal(0, 0.01, length)
        d["lam_L"] = np.linspace(0.6, 0.1, length) + rng.normal(0, 0.01, length)
        d["eps"] = np.linspace(0.5, 0.15, length) + rng.normal(0, 0.01, length)
    elif scenario == "willful_ignorance":
        d = base()
        d["lam_L"] = np.linspace(0.3, 0.8, length) + rng.normal(0, 0.01, length)
        d["xi"] = np.linspace(0.7, 0.15, length) + rng.normal(0, 0.01, length)
    elif scenario == "recovery_arc":
        third = length // 3
        d = {}
        for c in ALL_CONSTRUCTS:
            d[c] = np.concatenate([
                np.linspace(0.5, 0.15, third),
                np.full(third, 0.15),
                np.linspace(0.15, 0.45, length - 2 * third),
            ]) + rng.normal(0, 0.01, length)
        d["lam_L"] = np.concatenate([
            np.linspace(0.5, 0.15, third),
            np.full(third, 0.15),
            np.linspace(0.15, 0.7, length - 2 * third),
        ]) + rng.normal(0, 0.01, length)
    elif scenario == "sudden_crisis":
        d = base()
        cs, ce = length // 3, 2 * length // 3
        for c in ("kappa", "lam_P"):
            d[c][cs:ce] = np.linspace(0.5, 0.1, ce - cs) + rng.normal(0, 0.01, ce - cs)
    elif scenario == "slow_decay":
        rates = {"c": 0.002, "kappa": 0.001, "j": 0.003, "p": 0.001,
                 "eps": 0.002, "lam_L": 0.003, "lam_P": 0.001, "xi": 0.002}
        d = {c: 0.7 - rates[c] * np.arange(length) + rng.normal(0, 0.01, length) for c in ALL_CONSTRUCTS}
    elif scenario == "random_walk":
        d = {}
        for c in ALL_CONSTRUCTS:
            walk = np.zeros(length)
            walk[0] = 0.5
            for i in range(1, length):
                walk[i] = walk[i - 1] + rng.normal(0, 0.02)
                walk[i] += 0.01 * (0.5 - walk[i])
            d[c] = walk
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    df = pd.DataFrame(d)
    for c in ALL_CONSTRUCTS:
        df[c] = df[c].clip(0.0, 1.0)
    df["phi"] = df.apply(lambda row: compute_phi({c: row[c] for c in ALL_CONSTRUCTS}), axis=1)
    return df


# ============================================================================
# Inline model (CNN1D + LSTM + Attention + dual forecast heads)
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
# Training
# ============================================================================

def main():
    import trackio

    seed = 42
    pred_len = 10
    seq_len = 50
    hidden_size = 256
    epochs = 100
    batch_size = 64
    lr = 5e-4
    scenarios_per_type = 50
    length = 200
    window = 20

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    # --- Init Trackio ---
    token = os.environ.get("HF_TOKEN")
    trackio.init(
        repo_id="crichalchemist/phi-forecaster-training",
        token=token,
        auto_create=True,
    )

    # --- Generate data ---
    logger.info(f"Generating {len(SCENARIOS)} x {scenarios_per_type} scenarios...")
    all_X, all_y_phi, all_y_construct = [], [], []
    scaler = RobustScaler()
    feature_names = None

    for s_idx, scenario in enumerate(SCENARIOS):
        for i in range(scenarios_per_type):
            rng = np.random.default_rng(seed + s_idx * 1000 + i)
            df = generate_scenario(scenario, length=length, rng=rng)
            features = compute_all_signals(df, window=window)

            if feature_names is None:
                feature_names = list(features.columns)
                scaler.fit(features.values)

            X = scaler.transform(features[feature_names].values)
            n_features = X.shape[1]

            for j in range(len(X) - seq_len - pred_len):
                all_X.append(X[j:j + seq_len])
                phi_t = df["phi"].values[j + seq_len:j + seq_len + pred_len]
                if len(phi_t) < pred_len:
                    continue
                all_y_phi.append(phi_t.reshape(-1, 1))
                ct = np.stack([df[c].values[j + seq_len:j + seq_len + pred_len] for c in ALL_CONSTRUCTS], axis=-1)
                all_y_construct.append(ct)

    X_all = torch.tensor(np.array(all_X), dtype=torch.float32)
    y_phi = torch.tensor(np.array(all_y_phi), dtype=torch.float32)
    y_construct = torch.tensor(np.array(all_y_construct), dtype=torch.float32)
    logger.info(f"Data: {X_all.shape[0]} sequences, {n_features} features")

    # --- Split ---
    n = len(X_all)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_ds = torch.utils.data.TensorDataset(X_all[perm[:split]], y_phi[perm[:split]], y_construct[perm[:split]])
    val_ds = torch.utils.data.TensorDataset(X_all[perm[split:]], y_phi[perm[split:]], y_construct[perm[split:]])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, pin_memory=True)

    # --- Model ---
    model = PhiForecasterGPU(
        input_size=n_features, hidden_size=hidden_size, n_layers=2, pred_len=pred_len
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info(f"Model: {model.num_parameters:,} parameters")
    logger.info(f"Training {epochs} epochs, batch_size={batch_size}")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x, yp, yc = [b.to(device, non_blocking=True) for b in batch]
            optimizer.zero_grad()
            pp, cp, _ = model(x)
            loss = model.compute_loss(pp, yp, cp, yc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, yp, yc = [b.to(device, non_blocking=True) for b in batch]
                pp, cp, _ = model(x)
                val_loss += model.compute_loss(pp, yp, cp, yc).item()
        val_loss /= len(val_loader)

        scheduler.step()

        # Log
        trackio.log({"train_loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]})
        logger.info(f"Epoch {epoch}/{epochs} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/tmp/phi_forecaster_best.pt")

    # --- Save to Hub ---
    logger.info(f"Best val_loss: {best_val_loss:.6f}")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo("crichalchemist/phi-forecaster", exist_ok=True)
    api.upload_file(
        path_or_fileobj="/tmp/phi_forecaster_best.pt",
        path_in_repo="phi_forecaster_best.pt",
        repo_id="crichalchemist/phi-forecaster",
        commit_message=f"PhiForecaster checkpoint — val_loss={best_val_loss:.6f}",
    )
    # Save metadata
    metadata = {
        "model": "PhiForecasterGPU", "hidden_size": hidden_size, "n_layers": 2,
        "pred_len": pred_len, "input_features": n_features, "epochs": epochs,
        "best_val_loss": best_val_loss, "train_sequences": len(train_ds),
        "scenarios": list(SCENARIOS), "scenarios_per_type": scenarios_per_type,
    }
    with open("/tmp/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    api.upload_file(
        path_or_fileobj="/tmp/training_metadata.json",
        path_in_repo="training_metadata.json",
        repo_id="crichalchemist/phi-forecaster",
    )
    logger.info("Checkpoint + metadata pushed to crichalchemist/phi-forecaster")


if __name__ == "__main__":
    main()
