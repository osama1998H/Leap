"""
Adapt command implementation.

This module handles the 'adapt' CLI command for online model adaptation.
Provides a workflow for continuous model improvement based on recent trading data.

See ADR-0013 for design rationale.
"""

import logging
import os
from typing import TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    import argparse
    from config import SystemConfig
    from cli.system import LeapTradingSystem

from training.online_learning import AdaptiveTrainer, AdaptationConfig

logger = logging.getLogger(__name__)


def execute_adapt(
    system: 'LeapTradingSystem',
    args: 'argparse.Namespace',
    config: 'SystemConfig',
    resolved: dict
) -> None:
    """
    Execute the adapt command.

    Modes:
    - offline: Single adaptation pass with recent data
    - online: Continuous adaptation loop (experimental)
    - evaluate: Analyze current adaptation performance

    Args:
        system: LeapTradingSystem instance
        args: Parsed command-line arguments
        config: System configuration
        resolved: Resolved configuration values from CLI/config
    """
    symbols = resolved['symbols']
    timeframe = resolved['timeframe']
    n_bars = getattr(args, 'adapt_bars', 10000)
    mode = getattr(args, 'mode', 'offline')
    model_dir = getattr(args, 'model_dir', './saved_models')

    logger.info(f"Starting model adaptation (mode={mode})...")
    logger.info(f"Symbols: {symbols}, Timeframe: {timeframe}")

    # Load existing models
    try:
        system.load_models(model_dir)
    except Exception as e:
        logger.error(f"Failed to load models from {model_dir}: {e}")
        logger.info("Please train models first using: python main.py train")
        return

    if system._predictor is None or system._agent is None:
        logger.error("Models not loaded. Please train models first.")
        logger.info("Run: python main.py train --symbol <SYMBOL> --epochs 100 --timesteps 100000")
        return

    logger.info("Models loaded successfully")

    # Process each symbol
    for symbol in symbols:
        logger.info(f"{'='*50}")
        logger.info(f"Adapting models for {symbol}")
        logger.info(f"{'='*50}")

        # Load recent data for adaptation
        market_data = system.load_data(
            symbol=symbol,
            timeframe=timeframe,
            n_bars=n_bars
        )

        if market_data is None:
            logger.error(f"Failed to load data for {symbol}. Skipping...")
            continue

        logger.info(f"Loaded {len(market_data.close)} bars of data")

        # Create AdaptationConfig from CLI args
        adaptation_config = AdaptationConfig(
            error_threshold=getattr(args, 'error_threshold', 0.05),
            drawdown_threshold=getattr(args, 'drawdown_threshold', 0.1),
            adaptation_frequency=getattr(args, 'adapt_frequency', 100),
            max_adaptations_per_day=getattr(args, 'max_adaptations', 10),
            min_samples_for_adaptation=getattr(args, 'min_samples', 50)
        )

        # Execute based on mode
        if mode == 'offline':
            _run_offline_adaptation(
                system, args, config, market_data, adaptation_config, symbol, model_dir
            )
        elif mode == 'online':
            _run_online_adaptation(
                system, args, config, market_data, adaptation_config, symbol
            )
        elif mode == 'evaluate':
            _evaluate_adaptation(system, market_data, symbol, config)
        else:
            logger.error(f"Unknown mode: {mode}")


def _run_offline_adaptation(
    system: 'LeapTradingSystem',
    args: 'argparse.Namespace',
    config: 'SystemConfig',
    market_data,
    adaptation_config: AdaptationConfig,
    symbol: str,
    model_dir: str
) -> None:
    """
    Run single offline adaptation pass.

    This mode:
    1. Prepares training data from recent market data
    2. Performs a single training pass on both predictor and agent
    3. Saves the adapted models
    """
    import numpy as np
    from core.trading_env import TradingEnvironment
    from core.trading_types import EnvConfig

    logger.info("Running offline adaptation...")

    epochs = getattr(args, 'adapt_epochs', 10)
    timesteps = getattr(args, 'adapt_timesteps', 10000)

    # Prepare training data
    features = market_data.features
    if features is None:
        logger.error("No features in market data. Cannot adapt.")
        return

    n_samples = len(features)
    lookback = config.data.lookback_window

    if n_samples < lookback + 100:
        logger.error(f"Insufficient data for adaptation. Need at least {lookback + 100} samples.")
        return

    # Split into train/val (80/20)
    split_idx = int(n_samples * 0.8)

    # Prepare sequences for predictor
    X_train, y_train = _prepare_sequences(features[:split_idx], lookback, market_data.close[:split_idx])
    X_val, y_val = _prepare_sequences(features[split_idx:], lookback, market_data.close[split_idx:])

    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Create environment for agent adaptation
    env_config = EnvConfig(
        initial_balance=config.backtest.initial_balance,
        window_size=lookback
    )

    # Prepare OHLCV data as numpy array
    train_data = np.column_stack([
        market_data.open[:split_idx],
        market_data.high[:split_idx],
        market_data.low[:split_idx],
        market_data.close[:split_idx],
        market_data.volume[:split_idx]
    ])

    # Prepare features as numpy array
    train_features = features[:split_idx]

    env = TradingEnvironment(
        data=train_data,
        features=train_features,
        config=env_config
    )

    # Create adaptive trainer
    trainer = AdaptiveTrainer(
        predictor=system._predictor,
        agent=system._agent,
        env=env,
        config={'adaptation': adaptation_config.__dict__}
    )

    # Run offline training
    logger.info(f"Adapting predictor for {epochs} epochs...")
    logger.info(f"Adapting agent for {timesteps} timesteps...")

    try:
        results = trainer.train_offline(
            X_train, y_train,
            X_val, y_val,
            predictor_epochs=epochs,
            agent_timesteps=timesteps
        )

        _print_adaptation_results(results)

        # Save adapted models
        if getattr(args, 'save', False):
            # Create timestamped subdirectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            adapted_dir = os.path.join(model_dir, f"adapted_{timestamp}")
            trainer.save_models(adapted_dir)
            logger.info(f"Adapted models saved to {adapted_dir}")
        else:
            logger.info("Models not saved (use --save to persist adapted models)")

    except Exception as e:
        logger.error(f"Adaptation failed: {e}")
        raise


def _run_online_adaptation(
    system: 'LeapTradingSystem',
    args: 'argparse.Namespace',
    config: 'SystemConfig',
    market_data,
    adaptation_config: AdaptationConfig,
    symbol: str
) -> None:
    """
    Run continuous online adaptation loop.

    WARNING: This is experimental and requires careful monitoring.
    The adaptation loop will run until interrupted (Ctrl+C).
    """
    import time
    import numpy as np
    from core.trading_env import TradingEnvironment
    from core.trading_types import EnvConfig

    logger.warning("="*60)
    logger.warning("EXPERIMENTAL: Online adaptation mode")
    logger.warning("This will continuously adapt models based on market data.")
    logger.warning("Press Ctrl+C to stop.")
    logger.warning("="*60)

    features = market_data.features
    lookback = config.data.lookback_window

    # Create environment
    env_config = EnvConfig(
        initial_balance=config.backtest.initial_balance,
        window_size=lookback
    )

    # Prepare OHLCV data as numpy array
    data = np.column_stack([
        market_data.open,
        market_data.high,
        market_data.low,
        market_data.close,
        market_data.volume
    ])

    env = TradingEnvironment(
        data=data,
        features=features,
        config=env_config
    )

    # Create adaptive trainer
    trainer = AdaptiveTrainer(
        predictor=system._predictor,
        agent=system._agent,
        env=env,
        config={'adaptation': adaptation_config.__dict__}
    )

    # Create simple data stream wrapper
    class DataStream:
        def __init__(self, features, close, feature_names):
            self.features = features
            self.close = close
            self.feature_names = feature_names
            self.idx = lookback
            self.max_idx = len(close) - 1

        def get_latest(self):
            if self.idx >= self.max_idx:
                return None

            current_features = self.features[self.idx - lookback:self.idx]
            state = current_features.flatten()

            actual = (self.close[self.idx + 1] - self.close[self.idx]) / self.close[self.idx]

            result = {
                'features': current_features[-1],
                'state': state.astype('float32'),
                'close': self.close[self.idx],
                'returns': (self.close[self.idx] - self.close[self.idx - 1]) / self.close[self.idx - 1],
                'actual': actual,
                'reward': 0.0,
                'done': False
            }

            self.idx += 1
            return result

    data_stream = DataStream(features, market_data.close, market_data.feature_names)

    # Start online training
    def on_step(result):
        if result.get('adaptation_triggered'):
            logger.info(f"Adaptation triggered at step {result['step']}")
            logger.info(f"Regime: {result['regime']}")

    try:
        trainer.start_online_training(data_stream, callback=on_step)

        # Wait for completion or interrupt
        while trainer.is_training:
            time.sleep(1)

            # Check if data stream exhausted
            if data_stream.idx >= data_stream.max_idx:
                logger.info("Data stream exhausted. Stopping online training...")
                break

    except KeyboardInterrupt:
        logger.info("Online training interrupted by user")
    finally:
        trainer.stop_online_training()

        # Print final report
        report = trainer.online_manager.get_performance_report()
        _print_performance_report(report)


def _evaluate_adaptation(
    system: 'LeapTradingSystem',
    market_data,
    symbol: str,
    config: 'SystemConfig'
) -> None:
    """
    Evaluate current model adaptation performance.

    Provides metrics on prediction accuracy and agent performance
    on recent market data.
    """
    import numpy as np

    logger.info("Evaluating model adaptation performance...")

    features = market_data.features
    close = market_data.close
    lookback = config.data.lookback_window

    if features is None or len(features) < lookback + 10:
        logger.error("Insufficient data for evaluation")
        return

    # Evaluate predictor
    logger.info("\n--- Predictor Evaluation ---")
    predictions = []
    actuals = []

    for i in range(lookback, len(features) - 1):
        X = features[i - lookback:i].reshape(1, lookback, -1)

        try:
            pred = system._predictor.predict(X)
            predicted_return = pred['prediction'][0, 0]
            actual_return = (close[i + 1] - close[i]) / close[i]

            predictions.append(predicted_return)
            actuals.append(actual_return)
        except Exception as e:
            logger.debug(f"Prediction error at step {i}: {e}")
            continue

    if predictions:
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        correlation = np.corrcoef(predictions, actuals)[0, 1]

        # Direction accuracy
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        direction_acc = np.mean(pred_direction == actual_direction)

        logger.info(f"Samples evaluated: {len(predictions)}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"Correlation: {correlation:.4f}")
        logger.info(f"Direction Accuracy: {direction_acc:.2%}")

    # Evaluate agent
    logger.info("\n--- Agent Evaluation ---")
    from core.trading_types import Action

    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # HOLD, BUY, SELL, CLOSE

    for i in range(lookback, len(features)):
        obs = features[i - lookback:i].flatten().astype('float32')

        try:
            action, _, _ = system._agent.select_action(obs, deterministic=True)
            action_counts[int(action)] = action_counts.get(int(action), 0) + 1
        except Exception as e:
            logger.debug(f"Agent error at step {i}: {e}")
            continue

    total = sum(action_counts.values())
    if total > 0:
        logger.info(f"Total actions: {total}")
        for action, count in sorted(action_counts.items()):
            action_name = Action(action).name if action < 4 else f"ACTION_{action}"
            pct = count / total * 100
            logger.info(f"  {action_name}: {count} ({pct:.1f}%)")


def _prepare_sequences(features, lookback: int, close):
    """Prepare sequences for predictor training."""
    import numpy as np

    n_samples = len(features) - lookback - 1
    if n_samples <= 0:
        return np.array([]), np.array([])

    X = np.zeros((n_samples, lookback, features.shape[1]))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        X[i] = features[i:i + lookback]
        # Target: next return
        y[i] = (close[i + lookback + 1] - close[i + lookback]) / close[i + lookback]

    return X, y


def _print_adaptation_results(results: dict) -> None:
    """Print adaptation results summary."""
    logger.info("\n" + "="*50)
    logger.info("ADAPTATION RESULTS")
    logger.info("="*50)

    if 'predictor' in results:
        pred_results = results['predictor']
        logger.info("\nPredictor:")
        if isinstance(pred_results, dict):
            for key, value in pred_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.6f}")
                else:
                    logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  Final loss: {pred_results}")

    if 'agent' in results:
        agent_results = results['agent']
        logger.info("\nAgent:")
        if isinstance(agent_results, dict):
            for key, value in agent_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.6f}")
                else:
                    logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  Training result: {agent_results}")

    logger.info("="*50 + "\n")


def _print_performance_report(report: dict) -> None:
    """Print online learning performance report."""
    logger.info("\n" + "="*50)
    logger.info("ONLINE LEARNING PERFORMANCE REPORT")
    logger.info("="*50)

    if not report:
        logger.info("No performance data available")
        return

    logger.info(f"\nTotal steps: {report.get('total_steps', 0)}")
    logger.info(f"Total adaptations: {report.get('total_adaptations', 0)}")
    logger.info(f"Current regime: {report.get('current_regime', 'unknown')}")
    logger.info(f"Avg prediction error: {report.get('avg_prediction_error', 0):.6f}")
    logger.info(f"Total PnL: {report.get('total_pnl', 0):.2f}")
    logger.info(f"Avg Sharpe: {report.get('avg_sharpe', 0):.4f}")
    logger.info(f"Avg Win Rate: {report.get('avg_win_rate', 0):.2%}")

    if 'regime_distribution' in report:
        logger.info("\nRegime Distribution:")
        for regime, pct in report['regime_distribution'].items():
            logger.info(f"  {regime}: {pct:.1%}")

    logger.info("="*50 + "\n")


__all__ = ['execute_adapt']
