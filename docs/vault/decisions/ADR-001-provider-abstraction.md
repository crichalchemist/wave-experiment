---
id: ADR-001
title: ModelProvider Protocol abstraction
status: accepted
date: 2026-02-16
tags: [architecture, providers, protocol]
---

# ADR-001: ModelProvider Protocol Abstraction

## Decision

All model inference goes through a `@runtime_checkable` `ModelProvider` Protocol. Two concrete implementations: `VLLMProvider` (local Azure GPU VM) and `AzureFoundryProvider` (Claude via Azure AI Foundry).

## Context

The system needs to run two distinct model roles:
- **Generator**: local vLLM instance (DeepSeek-R1-Distill-Qwen-7B) for gap analysis
- **Critic**: Claude via Azure Foundry for CAI self-critique loops

These are not interchangeable — they are used for different pipeline stages. The abstraction makes the rest of the codebase independent of which concrete provider it receives.

## Consequences

- All training, inference, and hypothesis evolution code depends only on `ModelProvider`, never on a concrete class
- Provider is selected at startup via `provider_from_env()` reading `DETECTIVE_PROVIDER=vllm|azure`
- `provider_from_env()` fails fast if `DETECTIVE_PROVIDER` is not set — no silent defaults
- `MockProvider` is the standard test double throughout the test suite
- `VLLMProvider` and `AzureFoundryProvider` are both `frozen=True` dataclasses

## Files

- `src/core/providers.py` — Protocol, MockProvider, VLLMProvider, AzureFoundryProvider, provider_from_env
- `tests/core/test_providers.py`
