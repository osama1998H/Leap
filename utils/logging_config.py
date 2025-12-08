"""
Leap Trading System - Unified Logging Configuration
Centralized logging setup for consistent logging across all modules.
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional


# Default logging format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# File logging defaults
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    logs_dir: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    console_output: bool = True,
    auto_create_file: bool = True,
    log_filename_prefix: str = 'leap'
) -> logging.Logger:
    """
    Setup unified logging configuration for the Leap Trading System.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file path. If None, auto-generates based on logs_dir
        logs_dir: Directory for log files. If None, uses current directory
        log_format: Custom log format string
        date_format: Custom date format string
        max_bytes: Max size of log file before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 5)
        console_output: Whether to output logs to console
        auto_create_file: Auto-create log file if logs_dir is provided
        log_filename_prefix: Prefix for auto-generated log filenames (default 'leap')

    Returns:
        Root logger instance
    """
    # Use defaults if not specified
    log_format = log_format or DEFAULT_FORMAT
    date_format = date_format or DEFAULT_DATE_FORMAT

    # Get root logger
    root_logger = logging.getLogger()

    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    elif logs_dir and auto_create_file:
        # Auto-create log file with timestamp
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file_path = os.path.join(logs_dir, f'{log_filename_prefix}_{timestamp}.log')

        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This is the recommended way to get a logger in any module:
        from utils.logging_config import get_logger
        logger = get_logger(__name__)

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def add_file_handler(
    logger: logging.Logger,
    log_file: str,
    level: str = 'DEBUG',
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT
) -> logging.FileHandler:
    """
    Add a file handler to an existing logger.

    Useful for adding module-specific log files.

    Args:
        logger: Logger instance to add handler to
        log_file: Path to log file
        level: Logging level for this handler
        max_bytes: Max file size before rotation
        backup_count: Number of backup files

    Returns:
        The created file handler
    """
    # Ensure directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
    logger.addHandler(handler)

    return handler


class LogContext:
    """
    Context manager for temporarily changing log level.

    Usage:
        with LogContext(logger, 'DEBUG'):
            # Detailed logging here
            logger.debug("This will be logged")
    """

    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper(), logging.INFO)
        self.old_level = logger.level

    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
        return False
