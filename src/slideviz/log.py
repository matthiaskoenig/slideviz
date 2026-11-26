"""Logging setup shared by the slideviz entry points."""

from __future__ import annotations

import logging


class _Format(logging.Formatter):
    """Plain lines for progress, level and module for anything that went wrong."""

    def format(self, record: logging.LogRecord) -> str:
        """The bare message at INFO and below; level, logger name and traceback above."""
        if record.levelno <= logging.INFO:
            return record.getMessage()
        return super().format(record)


def setup(verbose: bool = False) -> None:
    """Send log messages to stderr, leaving stdout for data the caller may pipe."""
    handler = logging.StreamHandler()  # stderr by default
    handler.setFormatter(_Format("%(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=[handler],
    )
    # PyOpenGL reports its missing optional accelerator at info level on every start
    logging.getLogger("OpenGL").setLevel(logging.WARNING)
