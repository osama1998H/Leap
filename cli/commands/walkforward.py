"""
Walk-forward command implementation.

This module handles the 'walkforward' CLI command for running walk-forward
optimization and validation.
"""

import logging
import sys
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from config import SystemConfig
    from cli.system import LeapTradingSystem

logger = logging.getLogger(__name__)


def execute_walkforward(
    system: 'LeapTradingSystem',
    args: 'argparse.Namespace',
    config: 'SystemConfig',
    resolved: dict
) -> None:
    """
    Execute the walkforward command.

    Args:
        system: LeapTradingSystem instance
        args: Parsed command-line arguments
        config: System configuration
        resolved: Resolved configuration values
    """
    primary_symbol = resolved['primary_symbol']
    timeframe = resolved['timeframe']
    additional_timeframes = resolved['additional_timeframes']
    n_bars = resolved['n_bars']

    logger.info("Starting walk-forward analysis...")

    market_data = system.load_data(
        symbol=primary_symbol,
        timeframe=timeframe,
        n_bars=n_bars,
        additional_timeframes=additional_timeframes
    )

    if market_data is None:
        sys.exit(1)

    results = system.walk_forward_test(market_data)
    print(json.dumps(results, indent=2, default=str))
