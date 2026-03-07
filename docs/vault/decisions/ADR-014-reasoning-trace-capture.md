---
id: ADR-014
title: Reasoning trace capture and live display
status: accepted
date: 2026-03-03
tags: [architecture, traces, sse, observability, frontend]
---

# ADR-014: Reasoning Trace Capture and Live Display

## Decision

Capture chain-of-thought reasoning traces from the hybrid scoring provider, persist them to JSONL, stream them live via SSE, and display them on a dedicated `/reasoning` page at crichalchemist.com. The interception happens inside `HybridRoutingProvider.complete()` as a zero-call-site-change side-effect.

## Context

The DeepSeek-R1-Distill-Qwen-1.5B produces chain-of-thought reasoning before scoring ("Okay, so I need to evaluate... [analysis] ... score: 0.6"). All 6 scoring modules discard this reasoning — they regex-parse the float and throw away the text. This reasoning is valuable investigative context: it reveals how the model interprets evidence, what factors it weighs, and where it hedges.

Capturing these traces serves three purposes:

1. **Observability** — see what the scoring engine is thinking in real time
2. **Debugging** — when a score seems wrong, the CoT explains why
3. **Training data** — future fine-tuning can use CoT traces as supervision signal

## Implementation

### Data model

`ReasoningTrace` (frozen dataclass): id, timestamp, module, prompt, raw_response, parsed_score, model, route, duration_ms. A `ReasoningTrace.create()` static factory method auto-generates id (UUID), timestamp (UTC ISO), module (via `classify_module()`), and parsed_score (via `try_parse_score()`). Module classification uses heuristic matching of unique phrases in each module's prompt template. Score parsing extracts floats from "score: X" / "confidence: X" patterns.

### Storage

`TraceStore` -- append-only, thread-safe (Lock around file write + deque). JSONL on disk at `DETECTIVE_TRACE_PATH` for durability. Bounded deque (maxlen=500) for fast recent queries. `_load_recent_from_disk()` pre-populates the deque from the tail of the JSONL file on startup so `/traces/recent` is not cold after restart. `asyncio.Queue` subscriber list for SSE push.

### Provider integration

`HybridRoutingProvider._trace_store` field (default None). When set, `complete()` wraps the call with `time.monotonic()` timing and records a `ReasoningTrace`. Enabled by `DETECTIVE_TRACE_PATH` env var in `provider_from_env()`. Existing tests unchanged — store defaults to None.

### API endpoints

Three new routes in `create_app(trace_store=)`:
- `GET /traces/recent?n=50` — last N from deque
- `GET /traces/history?offset=0&limit=50` — paginated JSONL (newest first)
- `GET /traces/stream` — SSE with 30s keepalive comments

CORS middleware allows `crichalchemist.com` origins with `["GET", "POST"]` methods.

### Frontend

Self-contained HTML page at `/reasoning/index.html`. Two tabs: Live (SSE EventSource with auto-reconnect) and History (paginated fetch). Trace cards with collapsible CoT body, score coloring, prompt details. Green pulsing dot for connection status. DOM bounded to 200 cards.

### Deployment

Caddy reverse proxy: `/api/*` → `localhost:8080` with `flush_interval -1` for SSE. Systemd unit for uvicorn on `127.0.0.1:8080`.

## Consequences

- Zero changes to existing call sites — traces are captured transparently
- ~1KB per trace in JSONL (prompt + response + metadata)
- Deque bounds memory at 500 traces (~500KB)
- SSE keepalive prevents Cloudflare timeout on idle connections
- Thread-safe record() supports parallel_evolution's run_in_executor pattern

## Files

- `src/core/reasoning_trace.py` — dataclass + module classifier + score parser
- `src/core/trace_store.py` — JSONL persistence + deque + SSE subscribers
- `src/core/providers.py` — `_trace_store` field on HybridRoutingProvider
- `src/api/routes.py` — three trace endpoints + CORS
- `tests/core/test_reasoning_trace.py` — 16 tests
- `tests/core/test_trace_store.py` — 12 tests
- `tests/core/test_providers.py` — 4 new trace tests
- `tests/api/test_trace_routes.py` — 12 tests
