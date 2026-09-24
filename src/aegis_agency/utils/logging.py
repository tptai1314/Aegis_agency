"""Logging configuration for library and experiment code.

Library modules use module-level loggers (never ``print``); experiment scripts call
:func:`configure_logging` once at startup.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


class _StderrHandler(logging.StreamHandler):
    """StreamHandler that resolves ``sys.stderr`` at emit time.

    Binding the stream once (at configure time) breaks as soon as something replaces and closes
    ``sys.stderr`` — which is exactly what a test harness does with capture. Later records then
    fail with ``ValueError: I/O operation on closed file`` and the log line is lost. Resolving
    the stream per record keeps logging working under capture and after re-configuration.
    """

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        try:
            super().emit(record)
        except (ValueError, OSError):  # pragma: no cover - stream closed during teardown
            pass


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once, writing to stderr with a concise format."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = _StderrHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring a default handler exists."""
    if not _CONFIGURED:
        # Attach a single NullHandler so library import never emits noise; scripts call
        # configure_logging() to actually surface messages.
        root = logging.getLogger()
        if not any(isinstance(h, logging.NullHandler) for h in root.handlers):
            root.addHandler(logging.NullHandler())
    return logging.getLogger(name)
