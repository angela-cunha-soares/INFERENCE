"""Lightweight logging configuration for the bwb framework.

Avoids the verbosity (and dependency footprint) of structlog/loguru. The
module-level logger inherits the level from the ``BWB_LOG_LEVEL``
environment variable when :func:`configure_logging` is called.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_LOGGERS: dict[str, logging.Logger] = {}
_DEFAULT_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s :: %(message)s"


def configure_logging(
    level: Optional[str] = None,
    *,
    fmt: str = _DEFAULT_FORMAT,
    stream=None,
) -> None:
    """Idempotently configure the root logger for the framework."""
    resolved = (level or os.environ.get("BWB_LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, resolved, logging.INFO)

    root = logging.getLogger("bwb")
    root.setLevel(numeric)

    # Avoid stacking handlers if called twice
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str = "bwb") -> logging.Logger:
    """Return a cached child logger under the bwb namespace."""
    if name not in _LOGGERS:
        if not name.startswith("bwb"):
            name = f"bwb.{name}"
        _LOGGERS[name] = logging.getLogger(name)
    return _LOGGERS[name]
