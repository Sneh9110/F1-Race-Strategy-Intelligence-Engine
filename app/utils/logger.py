"""
Centralized Logging Utility with Structured Logging Support

Provides JSON-formatted logging with correlation IDs, log rotation, and monitoring integration.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import uuid


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation ID to log records."""

    def __init__(self):
        super().__init__()
        self.correlation_id = str(uuid.uuid4())

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to record."""
        if not hasattr(record, "correlation_id"):
            record.correlation_id = self.correlation_id
        return True


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_console: bool = True,
) -> None:
    """
    Setup centralized logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'text')
        log_file: Optional file path for log output
        enable_console: Whether to enable console output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Setup formatter
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(CorrelationIdFilter())
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(CorrelationIdFilter())
        root_logger.addHandler(file_handler)


def get_logger(name: str, extra_data: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger instance with optional extra data.

    Args:
        name: Logger name (typically __name__)
        extra_data: Optional dictionary of extra fields to include in logs

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if extra_data:
        # Create adapter to add extra data to all log calls
        logger = logging.LoggerAdapter(logger, {"extra_data": extra_data})

    return logger


def set_correlation_id(correlation_id: str) -> None:
    """
    Set correlation ID for current context.

    Args:
        correlation_id: Unique identifier for request tracing
    """
    # Update all handlers with new correlation ID
    for handler in logging.getLogger().handlers:
        for filter_obj in handler.filters:
            if isinstance(filter_obj, CorrelationIdFilter):
                filter_obj.correlation_id = correlation_id


# Example usage
if __name__ == "__main__":
    setup_logging(log_level="DEBUG", log_format="json")
    logger = get_logger(__name__, extra_data={"service": "f1-strategy"})

    logger.info("Application started")
    logger.debug("Debug information", extra={"session_id": "2024_MONACO_RACE"})
    logger.warning("High tire degradation detected")
    logger.error("Failed to fetch weather data", exc_info=True)
