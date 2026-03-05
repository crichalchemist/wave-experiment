---
id: ADR-019
title: Centralized logging convention
status: accepted
date: 2026-03-05
tags: [logging, observability, convention]
---

# ADR-019: Centralized Logging Convention

## Decision

Standardize all logging in the codebase to use `_logger = logging.getLogger(__name__)` (underscore-prefixed, module-private) and configure the `"src"` namespace logger once at startup via `src.core.log.configure_logging()`.

## Context

The codebase had three logging anti-patterns:
1. **Inconsistent naming** — some modules used `logger`, others `_logger`, most had none
2. **Silent failures** — 9 `except: pass` blocks swallowed errors with no diagnostic output
3. **Bypassed logging** — 20 `print(file=sys.stderr)` calls in training pipelines circumvented the logging hierarchy, making output uncontrollable by log level

These patterns made debugging production issues difficult — failures in source loading, model inference, and training pipelines produced no diagnostic output.

## Implementation

- `src/core/log.py` provides `configure_logging(level, fmt)` — idempotent, configures the `"src"` namespace handler
- All modules use `_logger = logging.getLogger(__name__)` (underscore prefix signals module-private by Python convention)
- `src/cli/main.py` calls `configure_logging()` in the CLI group entry point
- `src/api/routes.py` calls `configure_logging()` in `create_app()`
- Silent `except: pass` blocks replaced with `_logger.debug(...)` for diagnostic traceability
- `print(file=sys.stderr)` calls replaced with `_logger.info(...)` / `_logger.warning(...)`

## Consequences

- All log output flows through Python's logging hierarchy — controllable by level, format, and destination
- Debug-level messages available for troubleshooting without noise in production
- No duplicate handlers due to idempotent `configure_logging()`
- Consistent naming makes it easy to grep for logging usage across the codebase

## Files

- `src/core/log.py` (new)
- `src/cli/main.py` (modified — startup wiring)
- `src/api/routes.py` (modified — startup wiring)
- 8 files renamed `logger` → `_logger`
- 17 files added `_logger` where missing
- `src/training/constitutional_warmup.py` (modified — print→logging, except:pass→logging)
- `src/training/legal_warmup.py` (modified — print→logging)
- `src/data/sourcing/international_loader.py` (modified — except:pass→logging)
