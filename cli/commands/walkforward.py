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

    # Save data if requested
    if getattr(args, 'save_data', False):
        from utils.data_saver import save_pipeline_data, generate_run_id
        run_id = generate_run_id("walkforward", primary_symbol, timeframe)
        data_source = "MT5" if getattr(system.data_pipeline, 'broker_gateway', None) else "synthetic"
        save_pipeline_data(
            run_id=run_id,
            market_data=market_data,
            base_dir=config.get_path('data'),
            command="walkforward",
            n_bars=n_bars,
            data_source=data_source
        )
        logger.info(f"Pipeline data saved to {config.get_path('data')}/{run_id}/")

    results = system.walk_forward_test(market_data)
    print(json.dumps(results, indent=2, default=str))
