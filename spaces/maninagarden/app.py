"""
Phi Forecaster — Interactive Welfare Trajectory Demo.

Loads a CNN+LSTM+Attention checkpoint from crichalchemist/phi-forecaster
and lets users explore welfare trajectory forecasts via Gradio.

All model/signal/scenario code is copied inline from the training script
(scripts/train_phi_hf_job.py) to ensure exact state_dict compatibility.
"""
import math
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from huggingface_hub import hf_hub_download
import gradio as gr

# ============================================================================
# Inline welfare scoring (verbatim from train_phi_hf_job.py:39-75)
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
# Inline signal processing (verbatim from train_phi_hf_job.py:82-136)
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
    phi_vals = np.array([
        compute_phi({c: row[c] for c in ALL_CONSTRUCTS})
        for _, row in out[list(ALL_CONSTRUCTS)].iterrows()
    ])
    out["phi"] = phi_vals
    out["dphi_dt"] = np.gradient(phi_vals)
    return out


# ============================================================================
# Inline synthetic generator (verbatim from train_phi_hf_job.py:143-210)
# ============================================================================

SCENARIOS = (
    "stable_community", "capitalism_suppresses_love", "surveillance_state",
    "willful_ignorance", "recovery_arc", "sudden_crisis", "slow_decay", "random_walk",
)


def generate_scenario(scenario, length=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    def base(noise=0.01):
        return {c: 0.5 + rng.normal(0, noise, length).cumsum().clip(-0.1, 0.1)
                for c in ALL_CONSTRUCTS}

    if scenario == "stable_community":
        d = {c: 0.5 + rng.normal(0, 0.02, length).cumsum().clip(-0.1, 0.1)
             for c in ALL_CONSTRUCTS}
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
        d = {c: 0.7 - rates[c] * np.arange(length) + rng.normal(0, 0.01, length)
             for c in ALL_CONSTRUCTS}
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
    df["phi"] = df.apply(
        lambda row: compute_phi({c: row[c] for c in ALL_CONSTRUCTS}), axis=1
    )
    return df


# ============================================================================
# Inline model (verbatim from train_phi_hf_job.py:217-293)
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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

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
    def __init__(self, input_size=36, hidden_size=256, n_layers=2,
                 pred_len=10, dropout=0.1, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.cnn = CNN1D(input_size, hidden_size, kernel_size=3,
                         num_layers=2, dropout=dropout)
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
        return self._loss_fn(phi_pred, phi_target) + self.alpha * self._loss_fn(
            construct_pred, construct_target
        )

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Display constants
# ============================================================================

CONSTRUCT_DISPLAY = {
    "c": "Care (c)",
    "kappa": "Compassion (\u03ba)",
    "j": "Joy (j)",
    "p": "Purpose (p)",
    "eps": "Empathy (\u03b5)",
    "lam_L": "Love (\u03bb_L)",
    "lam_P": "Protection (\u03bb_P)",
    "xi": "Truth (\u03be)",
}

SCENARIO_DESCRIPTIONS = {
    "stable_community": "All constructs hover around 0.5 with low noise. "
                        "Phi remains stable. Baseline scenario.",
    "capitalism_suppresses_love": "Love (\u03bb_L) declines 0.6\u21920.1 as purpose erodes. "
                                  "Phi drops as community solidarity degrades. "
                                  "Material care persists but developmental support vanishes.",
    "surveillance_state": "Truth (\u03be) rises 0.3\u21920.9 while love drops 0.6\u21920.1. "
                          "Truth without love = surveillance. "
                          "Divergence penalty fires on (\u03bb_L \u2212 \u03be)\u00b2.",
    "willful_ignorance": "Love rises 0.3\u21920.8 while truth drops 0.7\u21920.15. "
                         "Love without truth = willful ignorance. "
                         "Community solidarity high but epistemic integrity collapses.",
    "recovery_arc": "All constructs drop to floor, then \u03bb_L recovers first. "
                    "Community leads recovery (Ubuntu). "
                    "Shows how love is the substrate for rebuilding.",
    "sudden_crisis": "Compassion (\u03ba) and protection (\u03bb_P) crash mid-scenario. "
                     "Crisis event disrupts emergency response and safeguarding.",
    "slow_decay": "All 8 constructs decline at different rates from 0.7. "
                  "Gradual institutional erosion \u2014 the boiling frog.",
    "random_walk": "Correlated random walks with mean-reversion around 0.5. "
                   "Tests general forecasting under uncertainty.",
}

CONSTRUCT_COLORS = [
    "#E91E63", "#9C27B0", "#FF9800", "#4CAF50",
    "#00BCD4", "#F44336", "#3F51B5", "#795548",
]

# ============================================================================
# Model loading (runs once at startup)
# ============================================================================

MODEL_REPO = "crichalchemist/phi-forecaster"
CHECKPOINT_FILE = "phi_forecaster_best.pt"
SEQ_LEN = 50
PRED_LEN = 10
INPUT_SIZE = 36
HIDDEN_SIZE = 256


def load_model():
    """Download checkpoint from Hub and load into PhiForecasterGPU."""
    path = hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILE)
    model = PhiForecasterGPU(
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
        n_layers=2, pred_len=PRED_LEN,
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"WARNING: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def build_reference_scaler():
    """Build RobustScaler from the same first scenario used in training."""
    rng = np.random.default_rng(42)
    df_ref = generate_scenario("stable_community", length=200, rng=rng)
    features_ref = compute_all_signals(df_ref, window=20)
    scaler = RobustScaler()
    scaler.fit(features_ref.values)
    return scaler, list(features_ref.columns)


print("Loading PhiForecaster checkpoint...")
MODEL = load_model()
print(f"Model loaded: {MODEL.num_parameters:,} parameters")

print("Building reference scaler...")
REFERENCE_SCALER, FEATURE_NAMES = build_reference_scaler()
print(f"Scaler ready: {len(FEATURE_NAMES)} features")


# ============================================================================
# Inference pipeline
# ============================================================================

def run_inference(df: pd.DataFrame):
    """Full inference pipeline: DataFrame -> (phi_pred, construct_pred, attention)."""
    features = compute_all_signals(df, window=20)
    X_scaled = REFERENCE_SCALER.transform(features[FEATURE_NAMES].values)
    X_window = X_scaled[-SEQ_LEN:]
    X_tensor = torch.tensor(X_window[np.newaxis], dtype=torch.float32)

    with torch.no_grad():
        phi_pred, construct_pred, attn_weights = MODEL(X_tensor)

    phi_trajectory = phi_pred[0, :, 0].numpy()
    construct_trajectories = construct_pred[0].numpy()
    attention = attn_weights[0].numpy()

    return phi_trajectory, construct_trajectories, attention


# ============================================================================
# Plotting helpers
# ============================================================================

def make_phi_plot(historical_phi, phi_forecast, title, length=200):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(length)), y=historical_phi,
        mode="lines", name="Historical \u03a6",
        line=dict(color="#2196F3", width=2),
    ))
    forecast_x = list(range(length, length + PRED_LEN))
    fig.add_trace(go.Scatter(
        x=forecast_x, y=phi_forecast,
        mode="lines+markers", name="Forecasted \u03a6",
        line=dict(color="#FF5722", width=3, dash="dash"),
        marker=dict(size=6),
    ))
    fig.add_vline(x=length - 0.5, line_dash="dot", line_color="gray",
                  annotation_text="Forecast horizon")
    fig.update_layout(
        title=title,
        xaxis_title="Time Step", yaxis_title="\u03a6 Score",
        yaxis_range=[0, 1], template="plotly_white", height=400,
    )
    return fig


def make_construct_plot(df, construct_forecast, title, length=200):
    fig = go.Figure()
    forecast_x = list(range(length, length + PRED_LEN))
    for i, c in enumerate(ALL_CONSTRUCTS):
        fig.add_trace(go.Scatter(
            x=list(range(length)), y=df[c].values,
            mode="lines", name=CONSTRUCT_DISPLAY[c],
            line=dict(color=CONSTRUCT_COLORS[i], width=1.5),
            legendgroup=c,
        ))
        fig.add_trace(go.Scatter(
            x=forecast_x, y=construct_forecast[:, i],
            mode="lines", name=f"{CONSTRUCT_DISPLAY[c]} (forecast)",
            line=dict(color=CONSTRUCT_COLORS[i], width=2.5, dash="dash"),
            legendgroup=c, showlegend=False,
        ))
    fig.add_vline(x=length - 0.5, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Time Step", yaxis_title="Construct Value [0, 1]",
        yaxis_range=[0, 1], template="plotly_white", height=500,
    )
    return fig


def make_attention_plot(attention):
    fig = go.Figure(go.Bar(
        x=list(range(len(attention))), y=attention,
        marker_color="#7C4DFF",
    ))
    fig.update_layout(
        title="Attention Weights (last 50 input steps)",
        xaxis_title="Input Step (within window)",
        yaxis_title="Attention Weight",
        template="plotly_white", height=300,
    )
    return fig


# ============================================================================
# Tab 1: Scenario Explorer
# ============================================================================

def explore_scenario(scenario: str, seed: int):
    rng = np.random.default_rng(int(seed))
    df = generate_scenario(scenario, length=200, rng=rng)

    phi_traj, construct_traj, attn = run_inference(df)

    fig_phi = make_phi_plot(
        df["phi"].values, phi_traj,
        f"\u03a6(humanity) \u2014 {scenario.replace('_', ' ').title()}",
    )
    fig_constructs = make_construct_plot(
        df, construct_traj,
        f"8 Welfare Constructs \u2014 {scenario.replace('_', ' ').title()}",
    )
    fig_attn = make_attention_plot(attn)

    description = SCENARIO_DESCRIPTIONS.get(scenario, "")
    return description, fig_phi, fig_constructs, fig_attn


# ============================================================================
# Tab 2: Custom Forecast
# ============================================================================

def generate_custom_scenario(levels: Dict[str, float], length=200, seed=42):
    rng = np.random.default_rng(seed)
    data = {}
    for c in ALL_CONSTRUCTS:
        center = levels[c]
        values = np.zeros(length)
        values[0] = center
        for i in range(1, length):
            reversion = 0.05 * (center - values[i - 1])
            noise = rng.normal(0, 0.015)
            values[i] = values[i - 1] + reversion + noise
        data[c] = np.clip(values, 0.0, 1.0)

    df = pd.DataFrame(data)
    df["phi"] = df.apply(
        lambda row: compute_phi({c: row[c] for c in ALL_CONSTRUCTS}), axis=1
    )
    return df


def custom_forecast(c_val, kappa_val, j_val, p_val, eps_val, lam_L_val, lam_P_val, xi_val):
    levels = {
        "c": c_val, "kappa": kappa_val, "j": j_val, "p": p_val,
        "eps": eps_val, "lam_L": lam_L_val, "lam_P": lam_P_val, "xi": xi_val,
    }

    current_phi = compute_phi(levels)
    df = generate_custom_scenario(levels)
    phi_traj, construct_traj, attn = run_inference(df)

    fig_phi = make_phi_plot(df["phi"].values, phi_traj, "Custom \u03a6 Trajectory")
    fig_constructs = make_construct_plot(
        df, construct_traj, "Custom Construct Trajectories"
    )

    phi_text = f"**Current \u03a6:** {current_phi:.3f}"
    return phi_text, fig_phi, fig_constructs


# ============================================================================
# Gradio UI
# ============================================================================

with gr.Blocks(
    title="\u03a6 Forecaster: Welfare Trajectory Demo",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("# \u03a6(humanity) Welfare Trajectory Forecaster")
    gr.Markdown(
        "Predict how community welfare evolves using a CNN+LSTM+Attention model "
        "trained on 8 synthetic scenarios. "
        "Grounded in care ethics (hooks 2000), capability theory (Sen 1999), "
        "and Ubuntu philosophy."
    )

    with gr.Tabs():
        # ==================== TAB 1: Scenario Explorer ====================
        with gr.Tab("Scenario Explorer"):
            with gr.Row():
                scenario_dropdown = gr.Dropdown(
                    choices=list(SCENARIOS),
                    value="stable_community",
                    label="Welfare Scenario",
                    info="Select a community trajectory archetype",
                )
                seed_slider = gr.Slider(
                    minimum=0, maximum=999, step=1, value=42,
                    label="Random Seed",
                    info="Varies noise realization within the same scenario",
                )
                forecast_btn = gr.Button("Run Forecast", variant="primary")

            scenario_description = gr.Markdown(
                value=SCENARIO_DESCRIPTIONS["stable_community"],
            )

            phi_plot = gr.Plot(label="\u03a6 Trajectory")
            construct_plot = gr.Plot(label="Construct Trajectories")
            attention_plot = gr.Plot(label="Attention Weights")

            forecast_btn.click(
                fn=explore_scenario,
                inputs=[scenario_dropdown, seed_slider],
                outputs=[scenario_description, phi_plot, construct_plot, attention_plot],
            )
            scenario_dropdown.change(
                fn=lambda s: SCENARIO_DESCRIPTIONS.get(s, ""),
                inputs=[scenario_dropdown],
                outputs=[scenario_description],
            )

        # ==================== TAB 2: Custom Forecast ====================
        with gr.Tab("Custom Forecast"):
            gr.Markdown("### Set construct levels and see how welfare evolves")
            gr.Markdown(
                "Generates a 200-step history around your chosen levels "
                "(with low noise + mean-reversion), then forecasts 10 steps ahead."
            )

            with gr.Row():
                c_slider = gr.Slider(
                    0.05, 0.95, value=0.5, step=0.05, label="Care (c)")
                kappa_slider = gr.Slider(
                    0.05, 0.95, value=0.5, step=0.05, label="Compassion (\u03ba)")
                j_slider = gr.Slider(
                    0.05, 0.95, value=0.5, step=0.05, label="Joy (j)")
                p_slider = gr.Slider(
                    0.05, 0.95, value=0.5, step=0.05, label="Purpose (p)")
            with gr.Row():
                eps_slider = gr.Slider(
                    0.05, 0.95, value=0.5, step=0.05, label="Empathy (\u03b5)")
                lam_L_slider = gr.Slider(
                    0.05, 0.95, value=0.5, step=0.05, label="Love (\u03bb_L)")
                lam_P_slider = gr.Slider(
                    0.05, 0.95, value=0.5, step=0.05, label="Protection (\u03bb_P)")
                xi_slider = gr.Slider(
                    0.05, 0.95, value=0.5, step=0.05, label="Truth (\u03be)")

            custom_btn = gr.Button("Generate & Forecast", variant="primary")
            phi_current_md = gr.Markdown("**Current \u03a6:** 0.500")
            custom_phi_plot = gr.Plot(label="\u03a6 Trajectory")
            custom_construct_plot = gr.Plot(label="Construct Trajectories")

            custom_btn.click(
                fn=custom_forecast,
                inputs=[c_slider, kappa_slider, j_slider, p_slider,
                        eps_slider, lam_L_slider, lam_P_slider, xi_slider],
                outputs=[phi_current_md, custom_phi_plot, custom_construct_plot],
            )

        # ==================== TAB 3: Research ====================
        with gr.Tab("Research"):
            gr.Markdown("""
## \u03a6(humanity): A Rigorous Ethical-Affective Objective Function

### The Formula

```
\u03a6(humanity) = f(\u03bb_L) \u00b7 [\u220f(\u0078\u0303_i)^(w_i)] \u00b7 \u03a8_ubuntu \u00b7 (1 \u2212 \u03a8_penalty)
```

**Components:**
- **f(\u03bb_L) = \u03bb_L^0.5** \u2014 Community solidarity multiplier (Ubuntu substrate). When community is low, all welfare degrades.
- **\u220f(x_i^w_i)** \u2014 Equity-weighted geometric mean (Nash Social Welfare Function). Any dimension \u2192 0 drives \u03a6 \u2192 0.
- **w_i = (1/x_i) / \u2211(1/x_j)** \u2014 Inverse-deprivation weights (Rawlsian maximin). Weights shift toward the most deprived construct.
- **\u03a8_ubuntu = 1 + 0.10\u00b7[\u221a(c\u00b7\u03bb_L) + \u221a(\u03ba\u00b7\u03bb_P) + \u221a(j\u00b7p) + \u221a(\u03b5\u00b7\u03be)] + 0.08\u00b7\u221a(\u03bb_L\u00b7\u03be)** \u2014 Relational synergy (Ubuntu: welfare emerges from relationships)
- **\u03a8_penalty = 0.15\u00b7[(c\u2212\u03bb_L)\u00b2 + (\u03ba\u2212\u03bb_P)\u00b2 + (j\u2212p)\u00b2 + (\u03b5\u2212\u03be)\u00b2 + (\u03bb_L\u2212\u03be)\u00b2] / 5** \u2014 Structural distortion penalty

### The Eight Constructs

| Symbol | Name | Definition | Floor | Citation |
|--------|------|-----------|-------|----------|
| c | Care | Resource allocation meeting basic needs | 0.20 | Tronto 1993 |
| \u03ba | Compassion | Responsive support to acute distress | 0.20 | Nussbaum 1996 |
| j | Joy | Positive affect above subsistence | 0.10 | Csikszentmihalyi 1990 |
| p | Purpose | Alignment of actions with chosen goals | 0.10 | Frankfurt 1971 |
| \u03b5 | Empathy | Accuracy of perspective-taking across groups | 0.10 | Batson 1991 |
| \u03bb_L | Love | Active extension for another's growth | 0.15 | hooks 2000 |
| \u03bb_P | Protection | Risk-weighted safeguarding from harm | 0.20 | Berlin 1969 |
| \u03be | Truth | Accuracy and transparency of records | 0.30 | Fricker 2007 |

### Synergy Pairs

| Pair | Constructs | Meaning | Distortion When Mismatched |
|------|-----------|---------|---------------------------|
| Care \u00d7 Love | c \u00b7 \u03bb_L | Material provision + developmental extension = true flourishing | Care without love = paternalistic control |
| Compassion \u00d7 Protection | \u03ba \u00b7 \u03bb_P | Emergency response + safeguarding = effective crisis intervention | Compassion without protection = vulnerable support |
| Joy \u00d7 Purpose | j \u00b7 p | Positive affect + goal-alignment = flow states | Joy without purpose = hedonic treadmill |
| Empathy \u00d7 Truth | \u03b5 \u00b7 \u03be | Perspective-taking + epistemic integrity = accurate understanding | Empathy without truth = manipulated solidarity |
| Love \u00d7 Truth | \u03bb_L \u00b7 \u03be | Investigative drive (curiosity) | Truth without love = surveillance; love without truth = willful ignorance |

### Model Architecture

**PhiForecasterGPU**: CNN1D (2-layer, kernel=3) \u2192 Stacked LSTM (256 hidden, 2 layers) \u2192 Additive Attention \u2192 Dual Heads:
- **\u03a6 Head**: 3-layer MLP \u2192 10 scalar predictions
- **Construct Head**: 3-layer MLP \u2192 10 \u00d7 8 construct predictions

**Training**: 8 scenarios \u00d7 50 instances, 100 epochs, val_loss = 0.000202

### Key References

- hooks, b. (2000). *All About Love: New Visions*. William Morrow.
- Sen, A. (1999). *Development as Freedom*. Oxford University Press.
- Fricker, M. (2007). *Epistemic Injustice: Power and the Ethics of Knowing*. Oxford University Press.
- Collins, P. H. (1990). *Black Feminist Thought*. Routledge.
- Nash, J. (1950). The bargaining problem. *Econometrica*, 18(2), 155-162.
- Rawls, J. (1971). *A Theory of Justice*. Harvard University Press.
- Nussbaum, M. (1996). Compassion: The basic social emotion. *Social Philosophy and Policy*.
- Csikszentmihalyi, M. (1990). *Flow: The Psychology of Optimal Experience*.
- Berlin, I. (1969). Two concepts of liberty. In *Four Essays on Liberty*.
- Metz, T. (2007). Toward an African moral theory. *Journal of Political Philosophy*.
- Ramose, M. B. (1999). *African Philosophy Through Ubuntu*. Mond Books.

### Limitations

- This is a **diagnostic tool**, not an optimization target (Goodhart's Law)
- The 8-construct taxonomy is Western-situated; requires adaptation for Ubuntu, Confucian, Buddhist, and Indigenous frameworks
- Trained on synthetic data only; real-world calibration pending
- Scaler is fit on reference scenario at startup (matching training conditions)

*\u03a6(humanity) is not a turnkey moral oracle. It is a disciplined framework forcing transparency about normative commitments while structurally preventing care-without-love dystopias and centering marginalized voices in the definition of flourishing.*
""")

    demo.launch(show_error=True)
