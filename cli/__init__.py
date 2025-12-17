"""
Leap Trading System - CLI Package

This package provides the command-line interface for the Leap Trading System.
All CLI logic has been modularized from the original main.py.

Usage:
    python main.py train --symbol EURUSD --epochs 100
    python main.py backtest --symbol EURUSD --realistic
    python main.py autotrade --paper
"""

# Suppress urllib3 SSL warning for LibreSSL compatibility (macOS)
# This must be done before any imports that load urllib3
import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass  # urllib3 < 2.0 doesn't have NotOpenSSLWarning

import logging

from .system import LeapTradingSystem
from .parser import create_parser, initialize_logging, resolve_cli_config
from .commands import execute_command

from utils.mlflow_tracker import MLFLOW_AVAILABLE

logger = logging.getLogger(__name__)


def main():
    """
    Main CLI entry point for the Leap Trading System.

    This function:
    1. Parses command-line arguments
    2. Loads and resolves configuration
    3. Initializes logging
    4. Creates the LeapTradingSystem
    5. Dispatches to the appropriate command handler
    """
    # Create and parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Resolve configuration (load configs, apply overrides)
    config, resolved = resolve_cli_config(args)

    # Setup logging with config and CLI overrides
    initialize_logging(
        config=config,
        log_level_override=args.log_level,
        log_file_override=args.log_file
    )

    # Log MLflow status
    if config.mlflow.enabled and MLFLOW_AVAILABLE:
        logger.info(f"MLflow tracking enabled: experiment='{config.mlflow.experiment_name}'")
    elif config.mlflow.enabled and not MLFLOW_AVAILABLE:
        logger.warning("MLflow is enabled in config but not installed. Install with: pip install mlflow")

    # Create system
    system = LeapTradingSystem(config)

    # Execute command
    execute_command(args.command, system, args, config, resolved)


__all__ = [
    'LeapTradingSystem',
    'main',
    'initialize_logging',
    'create_parser',
    'resolve_cli_config',
    'execute_command',
]
