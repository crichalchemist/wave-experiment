---
id: ADR-013
title: Hybrid provider routing — Ollama scoring + Azure reasoning
status: accepted
date: 2026-03-03
tags: [architecture, providers, routing, ollama, azure, inference]
---

# ADR-013: Hybrid Provider Routing

## Decision

Add a `HybridRoutingProvider` that inspects prompt text to route scoring calls to a local Ollama model (qwen2.5:0.5b) and reasoning calls to Azure Foundry (Claude). Prompt classification uses regex patterns matching the "Reply with ONLY: score:" directives already present in modules A/B/C, evolution, and graph scoring prompts. A circuit-breaker falls back to Azure when Ollama is unreachable.

## Context

The detective system runs on a CPU-only Azure VM. Earlier profiling (Session #S14) showed local vLLM with deepseek-r1:7b took 60s+ per call — unusable. Azure Foundry works but costs money per call. Analysis of all 15 `provider.complete()` call sites revealed two distinct categories:

1. **Scoring prompts** (6 call sites): Constrained output — "Reply with ONLY: score: \<float\>". A 0.5B model handles these adequately in 2-4s on CPU.
2. **Reasoning prompts** (9 call sites): Complex analysis, step-by-step reasoning, critique. Requires a capable model.

Routing at the provider level (not the call site) means zero changes to existing code — all 15 call sites continue calling `provider.complete(prompt)` unchanged.

## Implementation

- `classify_prompt(prompt)` matches 4 regex patterns against prompt text, returns `"scoring"` or `"reasoning"`
- `OllamaProvider` — frozen dataclass, 30s timeout (fail fast), defaults to `localhost:11434/v1` and `qwen2.5:0.5b`
- `HybridRoutingProvider` — mutable dataclass wrapping `OllamaProvider` + `AzureFoundryProvider`
- Circuit breaker: on first Ollama failure, `_ollama_available` flips `False`; all subsequent scoring calls route to Azure until `reset_fallback()` is called
- `provider_from_env()` supports `DETECTIVE_PROVIDER=hybrid` (also `ollama` for standalone local use)

## Consequences

- **No call-site changes**: All modules, pipeline, CLI, training scripts unchanged
- **New env var**: `DETECTIVE_PROVIDER=hybrid` activates routing; `OLLAMA_BASE_URL` and `OLLAMA_MODEL` optionally override defaults
- **Cost reduction**: Scoring calls (most frequent) are free/local; only reasoning calls hit Azure
- **Graceful degradation**: If Ollama dies mid-session, circuit breaker routes everything to Azure automatically
- **New dependency**: Ollama must be running locally for scoring; without it, falls back to Azure-only after first failure

## Files

- `src/core/providers.py` — `OllamaProvider`, `HybridRoutingProvider`, `classify_prompt()`, updated `provider_from_env()`
- `tests/core/test_providers.py` — 37 tests covering all providers, classification, routing, fallback, factory
