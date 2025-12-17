"""
Evaluate command implementation.

This module handles the 'evaluate' CLI command for evaluating trained models.
"""

import logging
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import argparse
    from config import SystemConfig
    from cli.system import LeapTradingSystem

logger = logging.getLogger(__name__)


def execute_evaluate(
    system: 'LeapTradingSystem',
    args: 'argparse.Namespace',
    config: 'SystemConfig',
    resolved: dict
) -> None:
    """
    Execute the evaluate command.

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

    logger.info("Evaluating models...")

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
        run_id = generate_run_id("evaluate", primary_symbol, timeframe)
        data_source = "MT5" if getattr(system.data_pipeline, 'broker_gateway', None) else "synthetic"
        save_pipeline_data(
            run_id=run_id,
            market_data=market_data,
            base_dir=config.get_path('data'),
            command="evaluate",
            n_bars=n_bars,
            data_source=data_source
        )
        logger.info(f"Pipeline data saved to {config.get_path('data')}/{run_id}/")

    system.load_models(args.model_dir)

    # Run evaluation
    env = system.create_environment(market_data)

    if system._predictor and system._agent:
        splits, _ = system.prepare_training_data(market_data)

        # Evaluate predictor
        predictions = system._predictor.predict(splits['test'][0])
        mae = np.mean(np.abs(predictions['prediction'].flatten() - splits['test'][1]))
        print(f"Predictor MAE: {mae:.6f}")

        # Evaluate agent
        avg_reward = system._agent.evaluate(env, n_episodes=10)
        print(f"Agent Avg Reward: {avg_reward:.2f}")
    else:
        logger.warning("Both predictor and agent required for evaluation")
