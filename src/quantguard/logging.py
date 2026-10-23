"""Structured logs for pipeline stages."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


PIPELINE_EVENTS = (
    "EVALUATION_STARTED",
    "EVALUATION_COMPLETED",
    "REGRESSION_DETECTED",
    "POLICY_FAILED",
    "GATE_PASSED",
    "GATE_FAILED",
    "GATE_ERROR",
    "GATE_WARNING",
    "REPORT_GENERATED",
)


class _HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event", "")
        prefix = f"{event} " if event else ""
        message = super().format(record)
        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key
            not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "event",
                "taskName",
            }
            and not key.startswith("_")
        }
        if extra:
            return f"{record.levelname} {prefix}{message} {json.dumps(extra, default=str, sort_keys=True)}"
        return f"{record.levelname} {prefix}{message}"


def configure_logging(level: str = "INFO", structured: bool = False) -> None:
    root = logging.getLogger("quantguard")
    root.handlers.clear()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stderr)
    if structured:
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler.setFormatter(_HumanFormatter("%(asctime)s %(name)s %(message)s"))
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"quantguard.{name}")


def log_event(logger: logging.Logger, event: str, message: str, **fields: Any) -> None:
    if event not in PIPELINE_EVENTS:
        raise ValueError(f"Unknown pipeline event: {event}")
    logger.info(message, extra={"event": event, **fields})
