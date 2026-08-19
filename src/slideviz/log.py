"""One place to turn logging on, so the entry points agree on the format."""

from __future__ import annotations

import logging


class _Format(logging.Formatter):
    """Plain lines for progress, level and module for anything that went wrong."""

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno <= logging.INFO:  # a report should read as a report
            return record.getMessage()
        return f"{record.levelname} {record.name}: {record.getMessage()}"


def setup(verbose: bool = False) -> None:
    """Send log messages to stderr, leaving stdout for data the caller may pipe."""
    handler = logging.StreamHandler()  # stderr by default
    handler.setFormatter(_Format())
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=[handler],
    )
