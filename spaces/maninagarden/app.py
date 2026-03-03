"""
Phi Research Workbench -- Interactive welfare trajectory forecasting + training.

7 tabs: Scenario Explorer, Custom Forecast, Experiment Lab,
Training (gated), Data Workshop (gated), Research, Entity Network (gated).
"""
import os
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import gradio as gr
import spaces

from welfare import (
    compute_phi, ALL_CONSTRUCTS, CONSTRUCT_FLOORS, CONSTRUCT_DISPLAY,
    PENALTY_PAIRS, FORMULA_VERSION,
)
from model import load_model, list_checkpoint_versions, load_training_metadata, SEQ_LEN, PRED_LEN, MODEL_REPO
from scenarios import (
    generate_scenario, compute_all_signals, build_reference_scaler,
    SCENARIOS, SCENARIO_DESCRIPTIONS,
)
from training import get_training_script, launch_training_job, HARDWARE_OPTIONS
from graph_client import fetch_graph, invalidate_cache
from graph_analytics import (
    detect_communities, compute_centrality, compute_degree,
    compute_clustering, ego_subgraph, BACKEND as GRAPH_BACKEND,
)
import networkx as nx

# ============================================================================
# Startup
# ============================================================================

print(f"Formula version: {FORMULA_VERSION}")
print("Loading PhiForecaster checkpoint...")
MODEL = load_model()
print(f"Model loaded: {MODEL.num_parameters:,} parameters")

print("Building reference scaler...")
REFERENCE_SCALER, FEATURE_NAMES = build_reference_scaler()
print(f"Scaler ready: {len(FEATURE_NAMES)} features")

CONSTRUCT_COLORS = [
    "#E91E63", "#9C27B0", "#FF9800", "#4CAF50",
    "#00BCD4", "#F44336", "#3F51B5", "#795548",
]


# ============================================================================
# Inference pipeline
# ============================================================================

def _run_inference_core(df, model=None):
    """Core inference: DataFrame -> (phi_pred, construct_pred, attention).

    Undecorated so it can be called from within other @spaces.GPU functions
    (ZeroGPU decorators cannot nest).
    """
    if model is None:
        model = MODEL
    device = next(model.parameters()).device
    features = compute_all_signals(df, window=20)
    X_scaled = REFERENCE_SCALER.transform(features[FEATURE_NAMES].values)
    X_window = X_scaled[-SEQ_LEN:]
    X_tensor = torch.tensor(X_window[np.newaxis], dtype=torch.float32).to(device)

    with torch.no_grad():
        phi_pred, construct_pred, attn_weights = model(X_tensor)

    return (
        phi_pred[0, :, 0].cpu().numpy(),
        construct_pred[0].cpu().numpy(),
        attn_weights[0].cpu().numpy(),
    )


@spaces.GPU(duration=30)
def run_inference(df, model=None):
    """GPU-accelerated inference: DataFrame -> (phi_pred, construct_pred, attention)."""
    return _run_inference_core(df, model=model)


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
        title=title, xaxis_title="Time Step", yaxis_title="\u03a6 Score",
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
        title=title, xaxis_title="Time Step",
        yaxis_title="Construct Value [0, 1]",
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
# Tab handlers
# ============================================================================

def explore_scenario(scenario, seed):
    rng = np.random.default_rng(int(seed))
    df = generate_scenario(scenario, length=200, rng=rng)
    phi_traj, construct_traj, attn = run_inference(df)
    title = scenario.replace("_", " ").title()
    return (
        SCENARIO_DESCRIPTIONS.get(scenario, ""),
        make_phi_plot(df["phi"].values, phi_traj, f"\u03a6(humanity) \u2014 {title}"),
        make_construct_plot(df, construct_traj, f"8 Welfare Constructs \u2014 {title}"),
        make_attention_plot(attn),
    )


def custom_forecast(c_val, kappa_val, j_val, p_val, eps_val, lam_L_val, lam_P_val, xi_val):
    levels = {
        "c": c_val, "kappa": kappa_val, "j": j_val, "p": p_val,
        "eps": eps_val, "lam_L": lam_L_val, "lam_P": lam_P_val, "xi": xi_val,
    }
    current_phi = compute_phi(levels)
    rng = np.random.default_rng(42)
    data = {}
    for c in ALL_CONSTRUCTS:
        center = levels[c]
        values = np.zeros(200)
        values[0] = center
        for i in range(1, 200):
            values[i] = values[i-1] + 0.05*(center - values[i-1]) + rng.normal(0, 0.015)
        data[c] = np.clip(values, 0.0, 1.0)
    df = pd.DataFrame(data)
    phi_vals = []
    for idx in range(len(df)):
        m = {c: df[c].iloc[idx] for c in ALL_CONSTRUCTS}
        derivs = {}
        if idx > 0:
            derivs = {c: float(df[c].iloc[idx] - df[c].iloc[idx-1]) for c in ALL_CONSTRUCTS}
        phi_vals.append(compute_phi(m, derivatives=derivs))
    df["phi"] = phi_vals

    phi_traj, construct_traj, _ = run_inference(df)
    return (
        f"**Current \u03a6:** {current_phi:.3f}",
        make_phi_plot(df["phi"].values, phi_traj, "Custom \u03a6 Trajectory"),
        make_construct_plot(df, construct_traj, "Custom Construct Trajectories"),
    )


def check_admin_key(key):
    """Verify admin key and return visibility updates."""
    expected = os.environ.get("ADMIN_KEY", "")
    if key == expected and expected:
        return gr.update(visible=True), "Unlocked."
    return gr.update(visible=False), "Invalid key."


def do_launch_training(key, epochs, lr, hidden_size, batch_size, scenarios_per_type, hardware):
    expected = os.environ.get("ADMIN_KEY", "")
    if key != expected or not expected:
        return "Not authenticated."
    job_id, msg = launch_training_job(
        epochs=int(epochs), lr=float(lr), hidden_size=int(hidden_size),
        batch_size=int(batch_size), scenarios_per_type=int(scenarios_per_type),
        hardware=hardware,
    )
    return msg


def inspect_data(key, scenario, seed):
    """Data Workshop: show raw trajectory + signal features."""
    expected = os.environ.get("ADMIN_KEY", "")
    if key != expected or not expected:
        return None, None, None, "Not authenticated."
    rng = np.random.default_rng(int(seed))
    df = generate_scenario(scenario, length=200, rng=rng)
    features = compute_all_signals(df, window=20)

    # Raw trajectory plot
    fig_raw = go.Figure()
    for i, c in enumerate(ALL_CONSTRUCTS):
        fig_raw.add_trace(go.Scatter(
            x=list(range(200)), y=df[c].values,
            mode="lines", name=CONSTRUCT_DISPLAY[c],
            line=dict(color=CONSTRUCT_COLORS[i]),
        ))
    fig_raw.update_layout(
        title=f"Raw Trajectory \u2014 {scenario.replace('_', ' ').title()}",
        xaxis_title="Time Step", yaxis_title="Value [0, 1]",
        yaxis_range=[0, 1], template="plotly_white", height=400,
    )

    # Feature heatmap (sampled columns for readability)
    sample_cols = [f"{c}_vol" for c in ALL_CONSTRUCTS[:4]] + [f"{c}_mom" for c in ALL_CONSTRUCTS[:4]] + ["phi", "dphi_dt"]
    fig_heat = go.Figure(go.Heatmap(
        z=features[sample_cols].values.T,
        x=list(range(200)),
        y=sample_cols,
        colorscale="RdBu_r",
    ))
    fig_heat.update_layout(
        title="Signal Features (sample)", xaxis_title="Time Step",
        template="plotly_white", height=400,
    )

    # Phi with recovery floors visible
    fig_phi = go.Figure()
    fig_phi.add_trace(go.Scatter(
        x=list(range(200)), y=df["phi"].values,
        mode="lines", name="\u03a6(humanity)",
        line=dict(color="#2196F3", width=2),
    ))
    fig_phi.update_layout(
        title="\u03a6 Trajectory (with recovery-aware floors)",
        xaxis_title="Time Step", yaxis_title="\u03a6",
        yaxis_range=[0, 1], template="plotly_white", height=300,
    )

    stats = f"**Features:** {features.shape[1]} columns, {features.shape[0]} rows\n"
    stats += f"**Phi range:** [{df['phi'].min():.3f}, {df['phi'].max():.3f}]\n"
    stats += f"**NaN count:** {features.isna().sum().sum()}"

    return fig_raw, fig_heat, fig_phi, stats


@spaces.GPU(duration=60)
def compare_experiments(scenario, seed, revision_a, revision_b):
    """Experiment Lab: run same scenario through two checkpoints (GPU-accelerated)."""
    rng_a = np.random.default_rng(int(seed))
    df = generate_scenario(scenario, length=200, rng=rng_a)

    try:
        model_a = load_model(revision=revision_a if revision_a != "latest" else None)
    except Exception as e:
        return None, None, f"Failed to load checkpoint A: {e}"
    try:
        model_b = load_model(revision=revision_b if revision_b != "latest" else None)
    except Exception as e:
        return None, None, f"Failed to load checkpoint B: {e}"

    phi_a, _, _ = _run_inference_core(df, model=model_a)
    phi_b, _, _ = _run_inference_core(df, model=model_b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(200)), y=df["phi"].values,
        mode="lines", name="Historical \u03a6", line=dict(color="#2196F3", width=2),
    ))
    forecast_x = list(range(200, 200 + PRED_LEN))
    fig.add_trace(go.Scatter(
        x=forecast_x, y=phi_a,
        mode="lines+markers", name=f"Checkpoint A ({revision_a[:8]})",
        line=dict(color="#FF5722", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_x, y=phi_b,
        mode="lines+markers", name=f"Checkpoint B ({revision_b[:8]})",
        line=dict(color="#4CAF50", width=2, dash="dot"),
    ))
    fig.add_vline(x=199.5, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"Checkpoint Comparison \u2014 {scenario.replace('_', ' ').title()}",
        xaxis_title="Time Step", yaxis_title="\u03a6",
        yaxis_range=[0, 1], template="plotly_white", height=450,
    )

    meta_a = load_training_metadata(revision=revision_a if revision_a != "latest" else None)
    meta_b = load_training_metadata(revision=revision_b if revision_b != "latest" else None)

    def fmt(m):
        if not m:
            return "No metadata available"
        val_loss = m.get('best_val_loss')
        val_str = f"{val_loss:.6f}" if isinstance(val_loss, (int, float)) else str(val_loss or '?')
        return (f"**Epochs:** {m.get('epochs', '?')} | "
                f"**Val Loss:** {val_str} | "
                f"**Formula:** {m.get('formula_version', 'unknown')}")

    meta_text = f"**A:** {fmt(meta_a)}\n\n**B:** {fmt(meta_b)}"
    return fig, meta_text, ""


# ============================================================================
# Entity Network handlers
# ============================================================================

# Community colors for network visualization (up to 20 communities)
_COMMUNITY_COLORS = [
    "#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#F44336", "#8BC34A", "#FF5722", "#3F51B5",
    "#CDDC39", "#795548", "#009688", "#FFC107", "#673AB7",
    "#03A9F4", "#FFEB3B", "#607D8B", "#E040FB", "#76FF03",
]


def _make_network_plot(nodes, edges, communities, centrality, title):
    """Build Plotly force-directed network visualization."""
    if not nodes:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white", height=600)
        fig.add_annotation(text="No graph data available", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font_size=16)
        return fig

    # Build networkx graph for layout computation
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in edges:
        G.add_edge(e["source"], e["target"])

    # Force-directed layout: k scales with graph size for readability
    k = 1.0 / max(1, len(nodes) ** 0.5)
    pos = nx.spring_layout(G, k=k, iterations=50, seed=42)

    # Edge traces
    edge_x, edge_y = [], []
    for e in edges:
        if e["source"] in pos and e["target"] in pos:
            x0, y0 = pos[e["source"]]
            x1, y1 = pos[e["target"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#ccc"),
        hoverinfo="none", showlegend=False,
    ))

    # Node traces: one per community for legend grouping
    max_pr = max(centrality.values()) if centrality else 1.0
    community_ids = sorted(set(communities.values())) if communities else [0]

    for comm_id in community_ids:
        comm_nodes = [n for n in nodes if communities.get(n, 0) == comm_id]
        if not comm_nodes:
            continue
        color = _COMMUNITY_COLORS[comm_id % len(_COMMUNITY_COLORS)]
        nx_list = [pos[n][0] for n in comm_nodes if n in pos]
        ny_list = [pos[n][1] for n in comm_nodes if n in pos]
        sizes = [
            max(6, 30 * (centrality.get(n, 0.0) / max(max_pr, 1e-9)))
            for n in comm_nodes if n in pos
        ]
        labels = [n for n in comm_nodes if n in pos]
        hover = [
            f"{n}<br>PageRank: {centrality.get(n, 0):.4f}<br>Community: {comm_id}"
            for n in comm_nodes if n in pos
        ]

        fig.add_trace(go.Scatter(
            x=nx_list, y=ny_list, mode="markers",
            marker=dict(size=sizes, color=color, line=dict(width=0.5, color="#333")),
            text=labels, hovertext=hover, hoverinfo="text",
            name=f"Community {comm_id}", legendgroup=f"c{comm_id}",
        ))

    fig.update_layout(
        title=title, showlegend=True, template="plotly_white", height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def network_overview(key):
    """Fetch graph, run analytics, render overview."""
    expected = os.environ.get("ADMIN_KEY", "")
    if key != expected or not expected:
        return None, "Not authenticated."

    data = fetch_graph()
    nodes, edges = data["nodes"], data["edges"]

    if not nodes:
        fig = go.Figure()
        fig.add_annotation(text="No graph data — is the detective API running?",
                           showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        fig.update_layout(template="plotly_white", height=600)
        return fig, "No graph data available."

    communities = detect_communities(nodes, edges)
    centrality = compute_centrality(nodes, edges)

    fig = _make_network_plot(
        nodes, edges, communities, centrality,
        f"Entity Network ({len(nodes)} nodes, {len(edges)} edges)",
    )

    # Stats sidebar
    n_communities = len(set(communities.values())) if communities else 0
    top_10 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_lines = "\n".join(
        f"| {name} | {score:.4f} |" for name, score in top_10
    )
    stats = (
        f"**Backend:** {GRAPH_BACKEND}\n\n"
        f"**Nodes:** {len(nodes)} | **Edges:** {len(edges)} | "
        f"**Communities:** {n_communities}\n\n"
        f"### Top 10 by PageRank\n\n"
        f"| Entity | PageRank |\n|--------|----------|\n{top_lines}"
    )
    return fig, stats


def network_explore(key, entity, hops):
    """Ego-subgraph exploration for a specific entity."""
    expected = os.environ.get("ADMIN_KEY", "")
    if key != expected or not expected:
        return None, "Not authenticated."

    if not entity or not entity.strip():
        return None, "Enter an entity name to explore."

    data = fetch_graph()
    nodes, edges = data["nodes"], data["edges"]

    if not nodes:
        return None, "No graph data available."

    ego = ego_subgraph(nodes, edges, center=entity.strip(), hops=int(hops))
    if not ego["nodes"]:
        return None, f"Entity **{entity}** not found in the graph."

    # Compute analytics on full graph for context
    communities = detect_communities(nodes, edges)
    centrality = compute_centrality(nodes, edges)
    degrees = compute_degree(nodes, edges)
    clustering = compute_clustering(nodes, edges)

    fig = _make_network_plot(
        ego["nodes"], ego["edges"], communities, centrality,
        f"{entity} — {int(hops)}-hop neighbourhood ({len(ego['nodes'])} nodes)",
    )

    # Entity details
    entity_clean = entity.strip()
    pr_rank = sorted(centrality.keys(), key=lambda n: centrality.get(n, 0), reverse=True)
    rank = pr_rank.index(entity_clean) + 1 if entity_clean in pr_rank else "?"

    # Direct connections
    direct = [e for e in edges if e["source"] == entity_clean or e["target"] == entity_clean]
    conn_lines = "\n".join(
        f"| {e['source']} | {e['target']} | {e['relation']} | {e['confidence']:.2f} |"
        for e in direct[:20]
    )

    details = (
        f"### {entity_clean}\n\n"
        f"**Degree:** {degrees.get(entity_clean, 0)} | "
        f"**PageRank:** {centrality.get(entity_clean, 0):.4f} (rank #{rank}) | "
        f"**Community:** {communities.get(entity_clean, '?')} | "
        f"**Clustering:** {clustering.get(entity_clean, 0):.3f}\n\n"
        f"### Direct Connections (up to 20)\n\n"
        f"| Source | Target | Relation | Confidence |\n"
        f"|--------|--------|----------|------------|\n{conn_lines}"
    )
    return fig, details


def refresh_graph_cache():
    """Invalidate cache and re-fetch."""
    invalidate_cache()
    data = fetch_graph()
    n = data["stats"]["node_count"]
    e = data["stats"]["edge_count"]
    return f"Cache refreshed. {n} nodes, {e} edges."


# ============================================================================
# Gradio UI
# ============================================================================

with gr.Blocks(
    title="\u03a6 Research Workbench",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(f"# \u03a6(humanity) Research Workbench")
    gr.Markdown(
        f"Formula **{FORMULA_VERSION}** | "
        "CNN+LSTM+Attention forecaster | "
        "Grounded in care ethics (hooks 2000), capability theory (Sen 1999), Ubuntu philosophy"
    )

    with gr.Tabs():
        # ==================== TAB 1: Scenario Explorer ====================
        with gr.Tab("Scenario Explorer"):
            with gr.Row():
                scenario_dropdown = gr.Dropdown(
                    choices=list(SCENARIOS), value="stable_community",
                    label="Welfare Scenario",
                )
                seed_slider = gr.Slider(0, 999, step=1, value=42, label="Random Seed")
                forecast_btn = gr.Button("Run Forecast", variant="primary")
            scenario_description = gr.Markdown(SCENARIO_DESCRIPTIONS["stable_community"])
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
                inputs=[scenario_dropdown], outputs=[scenario_description],
            )

        # ==================== TAB 2: Custom Forecast ====================
        with gr.Tab("Custom Forecast"):
            gr.Markdown("### Set construct levels and forecast")
            with gr.Row():
                c_slider = gr.Slider(CONSTRUCT_FLOORS["c"], 0.95, value=0.5, step=0.05, label="Care (c)")
                kappa_slider = gr.Slider(CONSTRUCT_FLOORS["kappa"], 0.95, value=0.5, step=0.05, label="Compassion (\u03ba)")
                j_slider = gr.Slider(CONSTRUCT_FLOORS["j"], 0.95, value=0.5, step=0.05, label="Joy (j)")
                p_slider = gr.Slider(CONSTRUCT_FLOORS["p"], 0.95, value=0.5, step=0.05, label="Purpose (p)")
            with gr.Row():
                eps_slider = gr.Slider(CONSTRUCT_FLOORS["eps"], 0.95, value=0.5, step=0.05, label="Empathy (\u03b5)")
                lam_L_slider = gr.Slider(CONSTRUCT_FLOORS["lam_L"], 0.95, value=0.5, step=0.05, label="Love (\u03bb_L)")
                lam_P_slider = gr.Slider(CONSTRUCT_FLOORS["lam_P"], 0.95, value=0.5, step=0.05, label="Protection (\u03bb_P)")
                xi_slider = gr.Slider(CONSTRUCT_FLOORS["xi"], 0.95, value=0.5, step=0.05, label="Truth (\u03be)")
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

        # ==================== TAB 3: Experiment Lab ====================
        with gr.Tab("Experiment Lab"):
            gr.Markdown("### Compare two checkpoints on the same scenario")
            with gr.Row():
                exp_scenario = gr.Dropdown(choices=list(SCENARIOS), value="stable_community", label="Scenario")
                exp_seed = gr.Slider(0, 999, step=1, value=42, label="Seed")
            with gr.Row():
                exp_rev_a = gr.Textbox(value="latest", label="Checkpoint A (commit SHA or 'latest')")
                exp_rev_b = gr.Textbox(value="latest", label="Checkpoint B (commit SHA or 'latest')")
            exp_btn = gr.Button("Compare", variant="primary")
            exp_plot = gr.Plot(label="Forecast Comparison")
            exp_meta = gr.Markdown("")
            exp_error = gr.Markdown("")

            exp_btn.click(
                fn=compare_experiments,
                inputs=[exp_scenario, exp_seed, exp_rev_a, exp_rev_b],
                outputs=[exp_plot, exp_meta, exp_error],
            )

        # ==================== TAB 4: Training (gated) ====================
        with gr.Tab("Training"):
            gr.Markdown("### Launch training on HF Jobs")
            with gr.Row():
                train_key = gr.Textbox(type="password", label="Admin Key", scale=2)
                train_unlock_btn = gr.Button("Unlock", scale=1)
            train_status = gr.Markdown("")

            with gr.Group(visible=False) as train_controls:
                with gr.Row():
                    train_epochs = gr.Slider(10, 500, value=100, step=10, label="Epochs")
                    train_lr = gr.Number(value=5e-4, label="Learning Rate")
                with gr.Row():
                    train_hidden = gr.Dropdown(choices=["64", "128", "256", "512"], value="256", label="Hidden Size")
                    train_batch = gr.Dropdown(choices=["16", "32", "64", "128"], value="64", label="Batch Size")
                with gr.Row():
                    train_scenarios = gr.Slider(10, 200, value=50, step=10, label="Scenarios per Type")
                    train_hardware = gr.Dropdown(choices=list(HARDWARE_OPTIONS.keys()), value="t4-small", label="Hardware")
                train_launch_btn = gr.Button("Launch Training Job", variant="primary")
                train_result = gr.Markdown("")

            train_unlock_btn.click(
                fn=check_admin_key, inputs=[train_key],
                outputs=[train_controls, train_status],
            )
            train_launch_btn.click(
                fn=do_launch_training,
                inputs=[train_key, train_epochs, train_lr, train_hidden,
                        train_batch, train_scenarios, train_hardware],
                outputs=[train_result],
            )

        # ==================== TAB 5: Data Workshop (gated) ====================
        with gr.Tab("Data Workshop"):
            gr.Markdown("### Inspect training data: scenarios, signals, features")
            with gr.Row():
                dw_key = gr.Textbox(type="password", label="Admin Key", scale=2)
                dw_unlock_btn = gr.Button("Unlock", scale=1)
            dw_status = gr.Markdown("")

            with gr.Group(visible=False) as dw_controls:
                with gr.Row():
                    dw_scenario = gr.Dropdown(choices=list(SCENARIOS), value="stable_community", label="Scenario")
                    dw_seed = gr.Slider(0, 999, step=1, value=42, label="Seed")
                dw_btn = gr.Button("Inspect", variant="primary")
                dw_raw = gr.Plot(label="Raw Trajectory")
                dw_heat = gr.Plot(label="Signal Features")
                dw_phi = gr.Plot(label="\u03a6 with Recovery Floors")
                dw_stats = gr.Markdown("")

            dw_unlock_btn.click(
                fn=check_admin_key, inputs=[dw_key],
                outputs=[dw_controls, dw_status],
            )
            dw_btn.click(
                fn=inspect_data, inputs=[dw_key, dw_scenario, dw_seed],
                outputs=[dw_raw, dw_heat, dw_phi, dw_stats],
            )

        # ==================== TAB 7: Entity Network (gated) ====================
        with gr.Tab("Entity Network"):
            gr.Markdown("### Knowledge graph exploration (detective API bridge)")
            with gr.Row():
                net_key = gr.Textbox(type="password", label="Admin Key", scale=2)
                net_unlock_btn = gr.Button("Unlock", scale=1)
            net_status = gr.Markdown("")

            with gr.Group(visible=False) as net_controls:
                with gr.Tabs():
                    with gr.Tab("Overview"):
                        with gr.Row():
                            net_refresh_btn = gr.Button("Refresh Graph", variant="secondary")
                            net_refresh_status = gr.Markdown("")
                        net_overview_btn = gr.Button("Load Network Overview", variant="primary")
                        net_overview_plot = gr.Plot(label="Entity Network")
                        net_overview_stats = gr.Markdown("")

                    with gr.Tab("Explorer"):
                        with gr.Row():
                            net_entity = gr.Textbox(label="Entity Name", placeholder="e.g. Jeffrey Epstein")
                            net_hops = gr.Slider(1, 4, step=1, value=2, label="Hops")
                            net_explore_btn = gr.Button("Explore", variant="primary")
                        net_explore_plot = gr.Plot(label="Ego Network")
                        net_explore_details = gr.Markdown("")

            net_unlock_btn.click(
                fn=check_admin_key, inputs=[net_key],
                outputs=[net_controls, net_status],
            )
            net_refresh_btn.click(
                fn=refresh_graph_cache, inputs=[],
                outputs=[net_refresh_status],
            )
            net_overview_btn.click(
                fn=network_overview, inputs=[net_key],
                outputs=[net_overview_plot, net_overview_stats],
            )
            net_explore_btn.click(
                fn=network_explore, inputs=[net_key, net_entity, net_hops],
                outputs=[net_explore_plot, net_explore_details],
            )

        # ==================== TAB 6: Research ====================
        with gr.Tab("Research"):
            gr.Markdown("""
## \u03a6(humanity): A Rigorous Ethical-Affective Objective Function

### The Formula (v""" + FORMULA_VERSION + """)

```
\u03a6(humanity) = f(\u03bb_L) \u00b7 [\u220f(x\u0303_i)^(w_i)] \u00b7 \u03a8_ubuntu \u00b7 (1 \u2212 \u03a8_penalty)
```

**Components:**
- **f(\u03bb_L) = \u03bb_L^0.5** \u2014 Community solidarity multiplier (Ubuntu substrate)
- **x\u0303_i = recovery_aware_input(x_i)** \u2014 Effective input with recovery-aware floors. Below-floor constructs receive community-mediated recovery potential.
- **w_i = (1/x\u0303_i) / \u2211(1/x\u0303_j)** \u2014 Inverse-deprivation weights (Rawlsian maximin)
- **\u03a8_ubuntu = 1 + 0.10\u00b7[\u221a(c\u00b7\u03bb_L) + \u221a(\u03ba\u00b7\u03bb_P) + \u221a(j\u00b7p) + \u221a(\u03b5\u00b7\u03be)] + 0.08\u00b7\u221a(\u03bb_L\u00b7\u03be)** \u2014 Relational synergy + curiosity
- **\u03a8_penalty = 0.15\u00b7[(c\u2212\u03bb_L)\u00b2 + (\u03ba\u2212\u03bb_P)\u00b2 + (j\u2212p)\u00b2 + (\u03b5\u2212\u03be)\u00b2 + (\u03bb_L\u2212\u03be)\u00b2] / 5** \u2014 Structural distortion penalty

### Recovery-Aware Floors (New)

When a construct falls below its hard floor, recovery depends on both **trajectory** (dx/dt) and **community capacity** (\u03bb_L):
- **Healing trajectory + strong community** \u2192 rapid recovery toward floor
- **Stagnant + strong community** \u2192 partial recovery (community compensates)
- **Stagnant + no community** \u2192 true collapse ("white supremacy signature")

*Key insight: care doesn't begin the uptick without community intervention.*

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
| Care \u00d7 Love | c \u00b7 \u03bb_L | Material provision + developmental extension | Care without love = paternalistic control |
| Compassion \u00d7 Protection | \u03ba \u00b7 \u03bb_P | Emergency response + safeguarding | Compassion without protection = vulnerable support |
| Joy \u00d7 Purpose | j \u00b7 p | Positive affect + goal-alignment | Joy without purpose = hedonic treadmill |
| Empathy \u00d7 Truth | \u03b5 \u00b7 \u03be | Perspective-taking + epistemic integrity | Empathy without truth = manipulated solidarity |
| Love \u00d7 Truth | \u03bb_L \u00b7 \u03be | Investigative drive (curiosity) | Truth without love = surveillance; love without truth = willful ignorance |

### Key References

- hooks, b. (2000). *All About Love: New Visions*. William Morrow.
- Sen, A. (1999). *Development as Freedom*. Oxford University Press.
- Fricker, M. (2007). *Epistemic Injustice*. Oxford University Press.
- Collins, P. H. (1990). *Black Feminist Thought*. Routledge.
- Metz, T. (2007). Toward an African moral theory. *Journal of Political Philosophy*.
- Ramose, M. B. (1999). *African Philosophy Through Ubuntu*. Mond Books.

### Limitations

- **Diagnostic tool**, not an optimization target (Goodhart's Law)
- 8-construct taxonomy is Western-situated; requires adaptation for Ubuntu, Confucian, Buddhist, and Indigenous frameworks
- Trained on synthetic data; real-world calibration pending
- Recovery-aware floors add theoretical richness but require retraining to reflect in model predictions

*\u03a6(humanity) is not a turnkey moral oracle. It is a disciplined framework forcing transparency about normative commitments.*
""")

    # Auto-run default scenario on page load
    demo.load(
        fn=explore_scenario,
        inputs=[scenario_dropdown, seed_slider],
        outputs=[scenario_description, phi_plot, construct_plot, attention_plot],
    )

    demo.launch(show_error=True)
