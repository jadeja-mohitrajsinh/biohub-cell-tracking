"""Structured logging helpers."""

from __future__ import annotations

import logging
from typing import Iterable

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress


def setup_logging(level: str = "INFO") -> None:
    """Configure rich-backed logging for CLI and library use."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)


def progress_bar(columns: Iterable[str] | None = None) -> Progress:
    """Return a configured Rich progress instance."""
    return Progress(*columns) if columns else Progress()


console = Console()
