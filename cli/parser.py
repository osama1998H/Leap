"""
CLI argument parser and logging initialization.

This module handles command-line argument parsing, configuration resolution,
and logging setup for the Leap Trading System.
"""

import argparse
import os
from typing import Optional, Tuple, Dict, Any, List

from config import (
    SystemConfig,
    get_config,
    load_training_config,
    load_data_config,
    load_backtest_config,
    load_risk_config,
    load_auto_trader_config,
    load_logging_config,
)
from utils.logging_config import setup_logging


def initialize_logging(
    config: SystemConfig,
    log_level_override: Optional[str] = None,
    log_file_override: Optional[str] = None
) -> None:
    """
    Initialize logging based on configuration and CLI overrides.

    Args:
        config: System configuration
        log_level_override: CLI override for log level
        log_file_override: CLI override for log file path
    """
    log_config = config.logging

    # CLI overrides take precedence
    level = log_level_override or log_config.level
    log_file = log_file_override

    # If no explicit log file but file logging is enabled, use logs_dir
    logs_dir = None
    if log_config.log_to_file and not log_file:
        logs_dir = os.path.join(config.base_dir, config.logs_dir)

    setup_logging(
        level=level,
        log_file=log_file,
        logs_dir=logs_dir,
        log_format=log_config.log_format,
        date_format=log_config.date_format,
        max_bytes=log_config.max_file_size_mb * 1024 * 1024,
        backup_count=log_config.backup_count,
        console_output=log_config.log_to_console,
        auto_create_file=log_config.log_to_file,
        log_filename_prefix=log_config.log_filename_prefix,
        rotation_type=log_config.rotation_type,
        rotation_when=log_config.rotation_when,
        rotation_interval=log_config.rotation_interval
    )


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all commands and options.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description='Leap Trading System - AI-Powered Forex Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --symbol EURUSD --epochs 100
  python main.py backtest --symbol EURUSD
  python main.py autotrade --paper
  python main.py evaluate --model-dir ./models
        """
    )

    parser.add_argument(
        'command',
        choices=['train', 'backtest', 'evaluate', 'walkforward', 'autotrade'],
        help='Command to execute'
    )

    parser.add_argument(
        '--symbol', '-s',
        default=None,
        help='Trading symbol (default: from config or EURUSD). For multi-symbol training, use --symbols'
    )

    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='Multiple symbols for training (e.g., --symbols EURUSD GBPUSD). Overrides --symbol'
    )

    parser.add_argument(
        '--timeframe', '-t',
        default=None,
        help='Primary timeframe (default: from config or 1h)'
    )

    parser.add_argument(
        '--multi-timeframe',
        action='store_true',
        help='Enable multi-timeframe features (uses additional_timeframes from config)'
    )

    parser.add_argument(
        '--bars', '-b',
        type=int,
        default=None,
        help='Number of bars to load (default: 50000)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Training epochs for predictor (default: from config or 100)'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Training timesteps for agent (default: from config or 100000)'
    )

    parser.add_argument(
        '--model-type',
        choices=['transformer', 'ppo', 'both'],
        default='both',
        help='Model type to train: transformer (predictor only), ppo (agent only), or both (default)'
    )

    parser.add_argument(
        '--model-dir', '-m',
        default='./saved_models',
        help='Directory for saving/loading models'
    )

    parser.add_argument(
        '--paper',
        action='store_true',
        help='Use paper trading mode'
    )

    parser.add_argument(
        '--realistic',
        action='store_true',
        help='Enable realistic backtesting constraints (limited trades, capped position size)'
    )

    parser.add_argument(
        '--monte-carlo',
        action='store_true',
        help='Run Monte Carlo simulation for risk analysis'
    )

    parser.add_argument(
        '--save-data',
        action='store_true',
        help='Save raw OHLCV and computed features to CSV files in data/{run_id}/'
    )

    # Modular config arguments
    parser.add_argument(
        '--training-config',
        help='Path to training config (transformer + ppo settings)'
    )

    parser.add_argument(
        '--data-config',
        help='Path to data config (symbols, timeframes, features)'
    )

    parser.add_argument(
        '--backtest-config',
        help='Path to backtest config (simulation settings)'
    )

    parser.add_argument(
        '--risk-config',
        help='Path to risk config (position sizing, limits)'
    )

    parser.add_argument(
        '--auto-trader-config',
        help='Path to auto-trader config'
    )

    parser.add_argument(
        '--logging-config',
        help='Path to logging config'
    )

    parser.add_argument(
        '--log-level', '-l',
        default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: from config or INFO)'
    )

    parser.add_argument(
        '--log-file',
        help='Log file path'
    )

    # MLflow arguments
    parser.add_argument(
        '--mlflow-experiment',
        default=None,
        help='MLflow experiment name (default: from config)'
    )

    parser.add_argument(
        '--mlflow-tracking-uri',
        default=None,
        help='MLflow tracking URI (default: sqlite:///mlflow.db)'
    )

    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking'
    )

    return parser


def resolve_cli_config(args: argparse.Namespace) -> Tuple[SystemConfig, Dict[str, Any]]:
    """
    Resolve CLI arguments with config files into final configuration.

    Args:
        args: Parsed command-line arguments

    Returns:
        Tuple of (SystemConfig, resolved_args dict with computed values)
    """
    # Start with default config
    config = get_config()

    # Load modular configs if provided
    if args.training_config:
        if os.path.exists(args.training_config):
            transformer_cfg, ppo_cfg, device, seed = load_training_config(args.training_config)
            config.transformer = transformer_cfg
            config.ppo = ppo_cfg
            config.device = device
            config.seed = seed
        else:
            print(f"Warning: Training config file not found: {args.training_config}")

    if args.data_config:
        if os.path.exists(args.data_config):
            config.data = load_data_config(args.data_config)
        else:
            print(f"Warning: Data config file not found: {args.data_config}")

    if args.backtest_config:
        if os.path.exists(args.backtest_config):
            config.backtest = load_backtest_config(args.backtest_config)
        else:
            print(f"Warning: Backtest config file not found: {args.backtest_config}")

    if args.risk_config:
        if os.path.exists(args.risk_config):
            config.risk = load_risk_config(args.risk_config)
        else:
            print(f"Warning: Risk config file not found: {args.risk_config}")

    if args.auto_trader_config:
        if os.path.exists(args.auto_trader_config):
            config.auto_trader = load_auto_trader_config(args.auto_trader_config)
        else:
            print(f"Warning: Auto-trader config file not found: {args.auto_trader_config}")

    if args.logging_config:
        if os.path.exists(args.logging_config):
            config.logging = load_logging_config(args.logging_config)
        else:
            print(f"Warning: Logging config file not found: {args.logging_config}")

    # Apply MLflow CLI overrides
    if args.no_mlflow:
        config.mlflow.enabled = False
    if args.mlflow_experiment:
        config.mlflow.experiment_name = args.mlflow_experiment
    if args.mlflow_tracking_uri:
        config.mlflow.tracking_uri = args.mlflow_tracking_uri

    # Resolve CLI defaults from config (CLI takes precedence over config)
    # Symbols: --symbols > --symbol > config.data.symbols[0] > 'EURUSD'
    if args.symbols:
        symbols = args.symbols
    elif args.symbol:
        symbols = [args.symbol]
    elif config.data.symbols:
        symbols = config.data.symbols
    else:
        symbols = ['EURUSD']

    # Primary symbol for single-symbol commands
    primary_symbol = symbols[0]

    # Timeframe: CLI > config > '1h'
    timeframe = args.timeframe or config.data.primary_timeframe or '1h'

    # Additional timeframes for multi-timeframe features
    additional_timeframes = config.data.additional_timeframes if args.multi_timeframe else []

    # Bars: CLI > 50000
    n_bars = args.bars if args.bars is not None else 50000

    # Epochs: CLI > config > 100
    epochs = args.epochs if args.epochs is not None else config.transformer.epochs

    # Timesteps: CLI > config > 100000
    timesteps = args.timesteps if args.timesteps is not None else config.ppo.total_timesteps

    resolved = {
        'symbols': symbols,
        'primary_symbol': primary_symbol,
        'timeframe': timeframe,
        'additional_timeframes': additional_timeframes,
        'n_bars': n_bars,
        'epochs': epochs,
        'timesteps': timesteps,
    }

    return config, resolved
