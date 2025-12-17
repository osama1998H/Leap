"""
Backtest command implementation.

This module handles the 'backtest' CLI command for running backtests
on historical data.
"""

import logging
import os
import sys
import json
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from config import SystemConfig
    from cli.system import LeapTradingSystem

logger = logging.getLogger(__name__)


def execute_backtest(
    system: 'LeapTradingSystem',
    args: 'argparse.Namespace',
    config: 'SystemConfig',
    resolved: dict
) -> None:
    """
    Execute the backtest command.

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

    logger.info("Starting backtest...")

    # Load data
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
        run_id = generate_run_id("backtest", primary_symbol, timeframe)
        data_source = "MT5" if getattr(system.data_pipeline, 'broker_gateway', None) else "synthetic"
        save_pipeline_data(
            run_id=run_id,
            market_data=market_data,
            base_dir=config.get_path('data'),
            command="backtest",
            n_bars=n_bars,
            data_source=data_source
        )
        logger.info(f"Pipeline data saved to {config.get_path('data')}/{run_id}/")

    # Load models if available
    if os.path.exists(args.model_dir):
        system.load_models(args.model_dir)

    # Run backtest
    _result, analysis = system.backtest(
        market_data,
        realistic_mode=args.realistic,
        enable_monte_carlo=args.monte_carlo
    )

    # Save results
    results_dir = os.path.join(config.base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results_data = {
        'total_return': analysis.get('total_return'),
        'sharpe_ratio': analysis.get('sharpe_ratio'),
        'max_drawdown': analysis.get('max_drawdown'),
        'win_rate': analysis.get('win_rate'),
        'total_trades': analysis.get('total_trades')
    }

    # Include Monte Carlo results if available
    if 'monte_carlo' in analysis:
        results_data['monte_carlo'] = analysis['monte_carlo']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(results_dir, f'backtest_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Log backtest results to MLflow
    tracker = system.mlflow_tracker
    if tracker and tracker.is_enabled:
        run_name = f"backtest-{primary_symbol}-{timeframe}-{timestamp}"
        with tracker.start_run(
            run_name=run_name,
            tags={
                "symbol": primary_symbol,
                "timeframe": timeframe,
                "command": "backtest",
                "realistic_mode": str(args.realistic)
            }
        ):
            # Log parameters
            tracker.log_params({
                "symbol": primary_symbol,
                "timeframe": timeframe,
                "n_bars": n_bars,
                "realistic_mode": args.realistic,
                "monte_carlo": args.monte_carlo
            })

            # Log backtest metrics
            tracker.log_backtest_results(analysis)

            # Log results file as artifact
            tracker.log_artifact(results_file)

            logger.info(f"Backtest results logged to MLflow: {run_name}")
