"""
Centralized logging configuration for the detective-llm project.

All modules should use: ``_logger = logging.getLogger(__name__)``
Startup entry points (CLI, API) call ``configure_logging()`` once.
"""
from __future__ import annotations

import logging

_DEFAULT_FMT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def configure_logging(
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FMT,
) -> None:
    """Configure the ``src`` namespace logger. Idempotent."""
    root = logging.getLogger("src")
    if root.handlers:
        return  # already configured
    root.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
