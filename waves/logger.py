"""
File: logger.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Logging utilities for the WAVES application.

License: MIT License
"""

import logging
from typing import Final

from waves.config import get_config_str

LOG_LEVELS: Final[dict[str, int]] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def get_logging_level() -> int:
    """Return the configured WAVES logging level."""

    level_name = get_config_str(
        "Logging_LEVEL",
        "INFO",
    ).upper()

    return LOG_LEVELS.get(level_name, logging.INFO)


def setup_logging() -> None:
    """Configure WAVES application logging."""

    logging_format = get_config_str(
        "Logging_FORMAT",
        "[%(levelname)s] %(name)s: %(message)s",
    )

    logging.basicConfig(
        level=get_logging_level(),
        format=logging_format,
        force=True,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger by name."""

    return logging.getLogger(name)
