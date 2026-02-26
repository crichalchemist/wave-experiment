---
title: Phi Forecaster
emoji: 🌊
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: mit
models:
  - crichalchemist/phi-forecaster
short_description: "Welfare trajectory forecasting with Φ(humanity)"
---

# Phi Forecaster: Welfare Trajectory Demo

Predict how community welfare evolves using the **Φ(humanity)** function — an ethical-affective objective function grounded in care ethics (hooks 2000), capability theory (Sen 1999), and Ubuntu philosophy.

## What This Does

A **CNN+LSTM+Attention** model trained on 8 synthetic welfare scenarios forecasts:
- **Φ trajectory** — composite welfare score over 10 future timesteps
- **8 construct trajectories** — care, compassion, joy, purpose, empathy, love, protection, truth
- **Attention weights** — which historical timesteps the model focused on

## The Formula

```
Φ(humanity) = f(λ_L) · [∏(x̃_i)^(w_i)] · Ψ_ubuntu · (1 − Ψ_penalty)
```

## Tabs

1. **Scenario Explorer** — Pick from 8 archetypal welfare trajectories and see forecasts
2. **Custom Forecast** — Set construct levels with sliders and forecast from there
3. **Research** — Full theoretical grounding, formula, citations

## Model

Checkpoint: [`crichalchemist/phi-forecaster`](https://huggingface.co/crichalchemist/phi-forecaster)

## Key References

- hooks, b. (2000). *All About Love: New Visions*
- Sen, A. (1999). *Development as Freedom*
- Fricker, M. (2007). *Epistemic Injustice*
- Collins, P. H. (1990). *Black Feminist Thought*
