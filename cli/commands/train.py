"""
Train command implementation.

This module handles the 'train' CLI command for training Transformer predictor
and/or PPO agent models.
"""

import logging
import os
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from config import SystemConfig
    from cli.system import LeapTradingSystem

logger = logging.getLogger(__name__)


def execute_train(
    system: 'LeapTradingSystem',
    args: 'argparse.Namespace',
    config: 'SystemConfig',
    resolved: dict
) -> None:
    """
    Execute the train command.

    Args:
        system: LeapTradingSystem instance
        args: Parsed command-line arguments
        config: System configuration
        resolved: Resolved configuration values
    """
    model_type = getattr(args, 'model_type', 'both')
    symbols = resolved['symbols']
    timeframe = resolved['timeframe']
    additional_timeframes = resolved['additional_timeframes']
    n_bars = resolved['n_bars']
    epochs = resolved['epochs']
    timesteps = resolved['timesteps']

    logger.info(f"Starting training (model_type={model_type})...")

    # Multi-symbol training
    if len(symbols) > 1:
        logger.info(f"Multi-symbol training enabled for: {symbols}")

    all_results = {}
    for symbol in symbols:
        logger.info(f"{'='*50}")
        logger.info(f"Training {model_type} on {symbol} ({timeframe})")
        logger.info(f"{'='*50}")

        # Load data with optional multi-timeframe features
        market_data = system.load_data(
            symbol=symbol,
            timeframe=timeframe,
            n_bars=n_bars,
            additional_timeframes=additional_timeframes
        )

        if market_data is None:
            logger.error(f"Failed to load data for {symbol}. Skipping...")
            continue

        # Route to appropriate training method based on model_type
        if model_type == 'transformer':
            results = system.train_predictor_only(
                market_data=market_data,
                predictor_epochs=epochs,
                symbol=symbol,
                timeframe=timeframe,
                additional_timeframes=additional_timeframes
            )
        elif model_type == 'ppo':
            results = system.train_agent_only(
                market_data=market_data,
                agent_timesteps=timesteps,
                symbol=symbol,
                timeframe=timeframe,
                additional_timeframes=additional_timeframes
            )
        else:  # 'both' - default behavior
            results = system.train(
                market_data=market_data,
                predictor_epochs=epochs,
                agent_timesteps=timesteps,
                symbol=symbol,
                timeframe=timeframe,
                additional_timeframes=additional_timeframes
            )

        all_results[symbol] = results

        # Save models per symbol if multi-symbol
        if len(symbols) > 1:
            symbol_model_dir = os.path.join(args.model_dir, symbol)
            if model_type == 'transformer':
                system._save_predictor_only(symbol_model_dir)
            elif model_type == 'ppo':
                system._save_agent_only(symbol_model_dir)
            else:
                system.save_models(symbol_model_dir)
            logger.info(f"Models for {symbol} saved to {symbol_model_dir}")

    # Save to default location for single symbol
    if len(symbols) == 1:
        if model_type == 'transformer':
            system._save_predictor_only(args.model_dir)
        elif model_type == 'ppo':
            system._save_agent_only(args.model_dir)
        else:
            system.save_models(args.model_dir)

    logger.info("Training complete!")

    # Print summary (handle partial training results)
    summary = {}
    for symbol, results in all_results.items():
        symbol_summary = {}
        if results.get('predictor') is not None:
            train_losses = results['predictor'].get('train_losses', [])
            symbol_summary['predictor_final_loss'] = train_losses[-1] if train_losses else None
        if results.get('agent') is not None:
            episode_rewards = results['agent'].get('episode_rewards', [])
            symbol_summary['agent_episodes'] = len(episode_rewards)
        summary[symbol] = symbol_summary
    print(json.dumps(summary, indent=2))
