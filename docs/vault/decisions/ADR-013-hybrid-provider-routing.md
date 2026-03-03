---
id: ADR-013
title: Hybrid provider routing — vLLM CPU scoring + Azure reasoning
status: accepted
date: 2026-03-03
tags: [architecture, providers, routing, vllm, azure, inference, docker]
---

# ADR-013: Hybrid Provider Routing

## Decision

Add a `HybridRoutingProvider` that inspects prompt text to route scoring calls to a local vLLM CPU instance (Qwen2.5-0.5B-Instruct via Docker) and reasoning calls to Azure Foundry (Claude). Prompt classification uses regex patterns matching the "Reply with ONLY: score:" directives already present in modules A/B/C, evolution, and graph scoring prompts. A circuit-breaker falls back to Azure when the scoring provider is unreachable.

## Context

The detective system runs on a CPU-only Azure VM (Intel Xeon Platinum 8370C, 4 cores, 62GB RAM, AVX-512 VNNI). Azure Foundry works but costs money per call. Analysis of all 15 `provider.complete()` call sites revealed two distinct categories:

1. **Scoring prompts** (6 call sites): Constrained output — "Reply with ONLY: score: \<float\>". A 0.5B model handles these adequately in ~700ms on CPU via vLLM.
2. **Reasoning prompts** (9 call sites): Complex analysis, step-by-step reasoning, critique. Requires a capable model.

Routing at the provider level (not the call site) means zero changes to existing code — all 15 call sites continue calling `provider.complete(prompt)` unchanged.

## Implementation

- `classify_prompt(prompt)` matches 4 regex patterns against prompt text, returns `"scoring"` or `"reasoning"`
- `HybridRoutingProvider` — mutable dataclass wrapping `VLLMProvider` (scoring) + `AzureFoundryProvider` (reasoning)
- Circuit breaker: on first scoring failure, `_scoring_available` flips `False`; all subsequent scoring calls route to Azure until `reset_fallback()` is called
- `provider_from_env()` supports `DETECTIVE_PROVIDER=hybrid`
- vLLM runs as a Docker container: `vllm/vllm-openai-cpu:latest-x86_64` on port 8100, serving `Qwen/Qwen2.5-0.5B-Instruct` with `--dtype float32`

## vLLM Docker setup

```bash
sudo docker run -d \
  --name vllm-cpu \
  --privileged --shm-size=4g \
  -p 8100:8000 \
  -e VLLM_CPU_KVCACHE_SPACE=4 \
  -e VLLM_CPU_OMP_THREADS_BIND=0-4 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai-cpu:latest-x86_64 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float32 \
  --max-model-len 512 \
  --host 0.0.0.0 --port 8000
```

## Consequences

- **No call-site changes**: All modules, pipeline, CLI, training scripts unchanged
- **New env vars**: `DETECTIVE_PROVIDER=hybrid` activates routing; `VLLM_SCORING_URL` and `VLLM_SCORING_MODEL` optionally override defaults (`http://localhost:8100/v1`, `Qwen/Qwen2.5-0.5B-Instruct`)
- **Cost reduction**: Scoring calls (most frequent) are free/local; only reasoning calls hit Azure
- **Graceful degradation**: If vLLM dies mid-session, circuit breaker routes everything to Azure automatically
- **Docker dependency**: vLLM CPU runs as a Docker container; must be started before hybrid mode

## Files

- `src/core/providers.py` — `HybridRoutingProvider`, `classify_prompt()`, updated `provider_from_env()`
- `tests/core/test_providers.py` — 32 tests covering all providers, classification, routing, fallback, factory
