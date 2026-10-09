"""Structured JSON Logging Configuration."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom formatter outputting logs as structured JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include contextual fields passed in extra
        if hasattr(record, "session_id"):
            log_record["session_id"] = record.session_id
        if hasattr(record, "event_type"):
            log_record["event_type"] = record.event_type
        if hasattr(record, "latency_ms"):
            log_record["latency_ms"] = record.latency_ms
        if hasattr(record, "model_name"):
            log_record["model_name"] = record.model_name
        if hasattr(record, "tool_name"):
            log_record["tool_name"] = record.tool_name
        if hasattr(record, "error"):
            log_record["error"] = str(record.error)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def configure_logging(level: int = logging.INFO) -> None:
    """Configures root logger with JSONFormatter writing to standard out."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance for the given namespace."""
    return logging.getLogger(name)
