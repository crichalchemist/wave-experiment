---
id: ADR-001
title: ModelProvider Protocol abstraction
status: accepted
date: 2026-02-16
tags: [architecture, providers, protocol]
---

# ADR-001: ModelProvider Protocol Abstraction

## Decision

All model inference goes through a `@runtime_checkable` `ModelProvider` Protocol. Three concrete implementations: `VLLMProvider` (local vLLM instance), `AzureFoundryProvider` (Claude via Azure AI Foundry), and `HybridRoutingProvider` (routes scoring to local vLLM, reasoning to Azure -- see ADR-013).

## Context

The system needs to run two distinct model roles:
- **Generator**: local vLLM instance (DeepSeek-R1-Distill-Qwen-7B) for gap analysis
- **Critic**: Claude via Azure Foundry for CAI self-critique loops

These are not interchangeable — they are used for different pipeline stages. The abstraction makes the rest of the codebase independent of which concrete provider it receives.

## Consequences

- All training, inference, and hypothesis evolution code depends only on `ModelProvider`, never on a concrete class
- Provider is selected at startup via `provider_from_env()` reading `DETECTIVE_PROVIDER=vllm|azure|hybrid`
- `provider_from_env()` fails fast if `DETECTIVE_PROVIDER` is not set -- no silent defaults
- `critic_provider_from_env()` loads a separate `AzureFoundryProvider` from `AZURE_CRITIC_*` env vars for CAI warmup (so local vLLM and Azure critique can be active simultaneously)
- `classify_prompt(prompt)` inspects prompt text against 4 regex patterns to classify as `"scoring"` or `"reasoning"` -- used by `HybridRoutingProvider` for routing
- `MockProvider` is the standard test double throughout the test suite; it is a plain `@dataclass` (not frozen)
- `VLLMProvider` and `AzureFoundryProvider` are both `frozen=True` dataclasses; `HybridRoutingProvider` is a mutable `@dataclass` (holds circuit-breaker state)
- `embed()` raises `NotImplementedError` on `AzureFoundryProvider` and `HybridRoutingProvider`

## Files

- `src/core/providers.py` -- Protocol, MockProvider, VLLMProvider, AzureFoundryProvider, HybridRoutingProvider, classify_prompt, provider_from_env, critic_provider_from_env
- `tests/core/test_providers.py`
