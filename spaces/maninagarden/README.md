---
title: Phi Forecaster
emoji: 🌊
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: mit
models:
  - crichalchemist/phi-forecaster
short_description: "Welfare trajectory forecasting with Φ(humanity)"
---

# Phi Research Workbench

Predict how community welfare evolves using the **Φ(humanity)** function — an ethical-affective objective function grounded in care ethics (hooks 2000), capability theory (Sen 1999), and Ubuntu philosophy.

## Tabs

1. **Scenario Explorer** — Pick from 8 archetypal welfare trajectories and see forecasts
2. **Custom Forecast** — Set construct levels with sliders and forecast from there
3. **Experiment Lab** — Compare checkpoints side-by-side on the same scenario
4. **Training** — Launch GPU training jobs on HF infrastructure (admin-gated)
5. **Data Workshop** — Inspect training data: raw trajectories, signal features, Φ computation (admin-gated)
6. **Research** — Full theoretical grounding, formula, citations

## Formula v2.1

```
Φ(humanity) = f(λ_L) · [∏(x̃_i)^(w_i)] · Ψ_ubuntu · (1 − Ψ_penalty)
```

Where x̃_i includes recovery-aware floors: below-floor constructs receive community-mediated recovery potential.

## Model

**PhiForecasterGPU**: CNN1D → Stacked LSTM → Additive Attention → Dual Heads

Checkpoint: [`crichalchemist/phi-forecaster`](https://huggingface.co/crichalchemist/phi-forecaster)

## Key References

- hooks, b. (2000). *All About Love: New Visions*
- Sen, A. (1999). *Development as Freedom*
- Fricker, M. (2007). *Epistemic Injustice*
- Collins, P. H. (1990). *Black Feminist Thought*
