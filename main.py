"""
Leap Trading System - Main Entry Point

This module is a backward-compatible wrapper that re-exports from the cli/ package.
All CLI logic has been moved to the cli/ package for better organization.

Usage:
    python main.py train --symbol EURUSD --epochs 100
    python main.py backtest --symbol EURUSD --realistic
    python main.py autotrade --paper

For development, you can also run:
    python -m cli
"""

# Suppress urllib3 SSL warning for LibreSSL compatibility (macOS)
# This must be done before any imports that load urllib3
import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass  # urllib3 < 2.0 doesn't have NotOpenSSLWarning

import sys
import os

# Add project root to path (backward compatibility)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Re-export everything from cli package for backward compatibility
# This ensures `from main import LeapTradingSystem` still works
from cli import LeapTradingSystem, main, initialize_logging

# Expose commonly patched imports at main module level for test compatibility
# Tests use patterns like: @patch('main.TransformerPredictor', MockPredictor)
from models.transformer import TransformerPredictor
from models.ppo_agent import PPOAgent
from training.trainer import ModelTrainer
from evaluation.backtester import Backtester
from evaluation.metrics import PerformanceAnalyzer
from config import get_config, SystemConfig

# Also expose config loaders for compatibility
from config import (
    load_training_config,
    load_data_config,
    load_backtest_config,
    load_risk_config,
    load_auto_trader_config,
    load_logging_config,
)


if __name__ == '__main__':
    main()
