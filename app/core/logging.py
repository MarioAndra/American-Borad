from __future__ import annotations

import logging
import sys
from typing import Any

_LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}


def configure_logging(level: str = "INFO") -> None:
    _configure_root(level)
    _configure_uvicorn_loggers(level)


def _configure_root(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(_LOG_LEVELS.get(level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_Formatter())
        root.addHandler(handler)


def _configure_uvicorn_loggers(level: str) -> None:
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.setLevel(_LOG_LEVELS.get(level.upper(), logging.INFO))
        logger.propagate = False
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(_Formatter())
            logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return (
            f"[{record.levelname}] {record.name} — {record.getMessage()}"
            f"{self._extra(record)}"
        )

    @staticmethod
    def _extra(record: logging.LogRecord) -> str:
        extras = {k: v for k, v in record.__dict__.items() if k not in logging.LogRecord(*[None] * 7).__dict__}
        return f" | extras={extras}" if extras else ""
