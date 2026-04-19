"""
Logging utilities for Causal-JEPA training.

Provides structured logging with:
- Console output with color-coded log levels
- File logging for persistent records
- Training metrics formatting
"""

import os
import sys
import logging
from typing import Optional


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_file: str = "training.log",
) -> logging.Logger:
    """
    Configure logging for the training pipeline.

    Sets up both console and file handlers with a consistent format.
    Console uses a compact format; file uses a detailed format.

    Args:
        log_dir: directory for log files (None = console only)
        log_level: logging level (default: INFO)
        log_file: filename for the log file

    Returns:
        The root logger configured for the project
    """
    root_logger = logging.getLogger("src")
    root_logger.setLevel(log_level)

    # Prevent duplicate handlers if called multiple times
    if root_logger.handlers:
        return root_logger

    # Console handler — compact format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler — detailed format
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, log_file),
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    return root_logger
