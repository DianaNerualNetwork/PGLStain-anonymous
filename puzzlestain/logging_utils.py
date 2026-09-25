"""Dependency-free logging configuration for PuzzleStain.

This module is deliberately named ``logging_utils`` (not ``logging``) so it does
not shadow the standard-library :mod:`logging` package. It provides
:func:`configure_logging`, which sets the ``puzzlestain`` *package* logger level
from config. Hydra's ``@hydra.main`` configures the ROOT logger, so the package
logger level must be set explicitly for the config knob to bite.

Only the standard library (:mod:`logging`) is used.
"""

from __future__ import annotations

import logging
from typing import Any

# The package logger whose level the config knob controls. All library loggers
# created with ``logging.getLogger(__name__)`` under ``puzzlestain.*`` inherit
# from this one, so setting its level is enough to gate emissions tree-wide.
PACKAGE_LOGGER_NAME = "puzzlestain"

logger = logging.getLogger(__name__)


def configure_logging(cfg: Any) -> None:
    """Apply the configured log level to the ``puzzlestain`` package logger.

    Accepts either a full config object exposing ``cfg.logging.level`` (the
    common case from ``train.py`` / ``cli.py``) or a bare level string /
    ``int``. Hydra's ``@hydra.main`` only configures the ROOT logger, so this
    explicitly sets the *package* logger level — that is what makes
    ``logging.level=DEBUG`` (or ``WARNING``) actually change what
    ``puzzlestain.*`` modules emit.

    Args:
        cfg: A resolved config object (with a ``logging.level`` attribute), or a
            level given directly as a string (e.g. ``"DEBUG"``) or an ``int``.
            An unknown / unparseable string falls back to ``INFO`` with a
            warning rather than raising.
    """
    level: Any = cfg
    logging_cfg = getattr(cfg, "logging", None)
    if logging_cfg is not None:
        level = getattr(logging_cfg, "level", "INFO")

    resolved: int
    if isinstance(level, str):
        mapped = logging.getLevelName(level.upper())
        if isinstance(mapped, int):
            resolved = mapped
        else:
            logger.warning("Unknown logging level %r; falling back to INFO", level)
            resolved = logging.INFO
    elif isinstance(level, int):
        resolved = level
    else:
        resolved = logging.INFO

    logging.getLogger(PACKAGE_LOGGER_NAME).setLevel(resolved)


__all__ = ["configure_logging", "PACKAGE_LOGGER_NAME"]
