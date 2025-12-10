"""
Leap Trading System - Main Orchestrator
Command-line interface and main entry point for the trading system.
"""

# Suppress urllib3 SSL warning for LibreSSL compatibility (macOS)
# This must be done before any imports that load urllib3
import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass  # urllib3 < 2.0 doesn't have NotOpenSSLWarning

import argparse
import sys
import os
import json
import time
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig, get_config
from core.data_pipeline import DataPipeline
from core.trading_env import TradingEnvironment
from core.risk_manager import RiskManager, DynamicRiskManager
from models.transformer import TransformerPredictor
from models.ppo_agent import PPOAgent
from training.trainer import ModelTrainer
from training.online_learning import OnlineLearningManager, AdaptiveTrainer
from evaluation.backtester import Backtester, WalkForwardOptimizer
from evaluation.metrics import PerformanceAnalyzer
import logging
from utils.logging_config import setup_logging
from utils.mlflow_tracker import MLflowTracker, create_tracker, MLFLOW_AVAILABLE

# Auto-trader imports (optional - may not be available on all platforms)
try:
    from core.mt5_broker import MT5BrokerGateway
    from core.auto_trader import AutoTrader
    from config.settings import AutoTraderConfig
    AUTO_TRADER_AVAILABLE = True
except ImportError:
    AUTO_TRADER_AVAILABLE = False

import numpy as np


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


logger = logging.getLogger(__name__)


class LeapTradingSystem:
    """
    Main orchestrator for the Leap Trading System.

    Coordinates all components:
    - Data pipeline
    - Prediction model (Transformer)
    - RL agent (PPO)
    - Risk management
    - Backtesting
    - Online learning
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or get_config()
        self._setup_directories()

        # Initialize components (lazy loading)
        self._data_pipeline = None
        self._predictor = None
        self._agent = None
        self._risk_manager = None
        self._trainer = None
        self._backtester = None
        self._online_manager = None
        self._mlflow_tracker = None

        # Feature names used for model training (for inference compatibility)
        self._model_feature_names = None

        logger.info("Leap Trading System initialized")

    @property
    def mlflow_tracker(self) -> Optional[MLflowTracker]:
        """Get or create MLflow tracker."""
        if self._mlflow_tracker is None and self.config.mlflow.enabled:
            self._mlflow_tracker = create_tracker(self.config)
        return self._mlflow_tracker

    def _setup_directories(self):
        """Create necessary directories."""
        dirs = ['models', 'logs', 'data', 'checkpoints', 'results']
        for d in dirs:
            path = os.path.join(self.config.base_dir, getattr(self.config, f'{d}_dir', d))
            os.makedirs(path, exist_ok=True)

    @property
    def data_pipeline(self) -> DataPipeline:
        """Get or create data pipeline."""
        if self._data_pipeline is None:
            self._data_pipeline = DataPipeline(self.config.data)
        return self._data_pipeline

    @property
    def predictor(self) -> Optional[TransformerPredictor]:
        """Get predictor (initialized via initialize_models or load_models)."""
        return self._predictor

    @property
    def agent(self) -> Optional[PPOAgent]:
        """Get RL agent (initialized via initialize_models or load_models)."""
        return self._agent

    @property
    def risk_manager(self) -> RiskManager:
        """Get or create risk manager."""
        if self._risk_manager is None:
            self._risk_manager = DynamicRiskManager(
                initial_balance=self.config.backtest.initial_balance,
            )
        return self._risk_manager

    def load_data(
        self,
        symbol: str = 'EURUSD',
        timeframe: str = '1h',
        n_bars: int = 50000,
        additional_timeframes: Optional[list] = None
    ):
        """Load and prepare market data with optional multi-timeframe features.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Primary timeframe (e.g., '1h')
            n_bars: Number of bars to load
            additional_timeframes: List of additional timeframes to use as features
                                   (e.g., ['15m', '4h', '1d'])

        Returns:
            MarketData object with features from all timeframes
        """
        logger.info(f"Loading data for {symbol} {timeframe}...")

        # Connect to data source
        self.data_pipeline.connect()

        # Fetch historical data with multi-timeframe features
        market_data = self.data_pipeline.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            n_bars=n_bars,
            additional_timeframes=additional_timeframes
        )

        if market_data is None:
            logger.error("Failed to load market data")
            return None

        n_features = len(market_data.feature_names) if market_data.feature_names else 0
        logger.info(f"Loaded {len(market_data.close)} bars with {n_features} features")

        if additional_timeframes:
            logger.info(f"Multi-timeframe features included from: {additional_timeframes}")

        return market_data

    def prepare_training_data(self, market_data):
        """Prepare data for model training."""
        logger.info("Preparing training sequences...")

        X, y, _timestamps = self.data_pipeline.prepare_sequences(
            data=market_data,
            sequence_length=self.config.data.lookback_window,
            prediction_horizon=self.config.data.prediction_horizon
        )

        logger.info(f"Created {len(X)} sequences with shape {X.shape}")

        # Split data
        splits = self.data_pipeline.create_train_val_test_split(
            X, y,
            train_ratio=self.config.data.train_test_split,
            val_ratio=self.config.data.validation_split
        )

        return splits, X.shape[2]  # Return input dimension

    def initialize_models(self, input_dim: int, state_dim: int):
        """Initialize prediction and RL models."""
        logger.info(f"Initializing models (input_dim={input_dim}, state_dim={state_dim})...")

        # Initialize predictor
        self._predictor = TransformerPredictor(
            input_dim=input_dim,
            config={
                'd_model': self.config.transformer.d_model,
                'n_heads': self.config.transformer.n_heads,
                'n_encoder_layers': self.config.transformer.n_encoder_layers,
                'd_ff': self.config.transformer.d_ff,
                'dropout': self.config.transformer.dropout,
                'max_seq_length': self.config.data.lookback_window,
                'learning_rate': self.config.transformer.learning_rate,
                'weight_decay': self.config.transformer.weight_decay,
                'online_learning_rate': self.config.transformer.online_learning_rate
            },
            device=self.config.device
        )

        # Initialize agent
        self._agent = PPOAgent(
            state_dim=state_dim,
            action_dim=4,  # HOLD, BUY, SELL, CLOSE
            config={
                'learning_rate': self.config.ppo.learning_rate,
                'gamma': self.config.ppo.gamma,
                'gae_lambda': self.config.ppo.gae_lambda,
                'clip_epsilon': self.config.ppo.clip_epsilon,
                'entropy_coef': self.config.ppo.entropy_coef,
                'value_coef': self.config.ppo.value_coef,
                'max_grad_norm': self.config.ppo.max_grad_norm,
                'n_steps': self.config.ppo.n_steps,
                'n_epochs': self.config.ppo.n_epochs,
                'batch_size': self.config.ppo.batch_size,
                'hidden_sizes': self.config.ppo.actor_hidden_sizes
            },
            device=self.config.device
        )

        logger.info("Models initialized successfully")

    def create_environment(self, market_data) -> TradingEnvironment:
        """Create trading environment from market data."""
        # Prepare OHLCV data
        ohlcv = np.column_stack([
            market_data.open,
            market_data.high,
            market_data.low,
            market_data.close,
            market_data.volume
        ])

        env = TradingEnvironment(
            data=ohlcv,
            features=market_data.features,
            initial_balance=self.config.backtest.initial_balance,
            commission=self.config.backtest.commission_per_lot / 100000,
            spread=self.config.backtest.spread_pips * 0.0001,
            slippage=self.config.backtest.slippage_pips * 0.0001,
            leverage=self.config.backtest.leverage,
            window_size=self.config.data.lookback_window
        )

        return env

    def train(
        self,
        market_data,
        predictor_epochs: Optional[int] = None,
        agent_timesteps: Optional[int] = None,
        symbol: str = 'EURUSD',
        timeframe: str = '1h',
        additional_timeframes: Optional[list] = None
    ):
        """Train both models.

        Args:
            market_data: Market data for training
            predictor_epochs: Number of epochs for predictor training
            agent_timesteps: Number of timesteps for agent training
            symbol: Trading symbol (for MLflow tracking)
            timeframe: Timeframe (for MLflow tracking)
            additional_timeframes: List of additional timeframes used for features
        """
        # Store feature names used during training for inference compatibility
        if market_data.feature_names:
            self._model_feature_names = list(market_data.feature_names)
            logger.info(f"Training with {len(self._model_feature_names)} computed features")

        # Prepare data
        splits, input_dim = self.prepare_training_data(market_data)

        # Create environment
        env = self.create_environment(market_data)
        state_dim = env.observation_space.shape[0]

        # Initialize models
        self.initialize_models(input_dim, state_dim)

        # Create trainer with MLflow tracker
        trainer = ModelTrainer(
            predictor=self._predictor,
            agent=self._agent,
            data_pipeline=self.data_pipeline,
            config={
                'predictor_epochs': predictor_epochs or self.config.transformer.epochs,
                'agent_timesteps': agent_timesteps or self.config.ppo.total_timesteps,
                'batch_size': self.config.transformer.batch_size,
                'patience': self.config.transformer.patience,
                'checkpoint_dir': os.path.join(self.config.base_dir, self.config.checkpoints_dir)
            },
            mlflow_tracker=self.mlflow_tracker
        )

        # Start MLflow run if enabled
        tracker = self.mlflow_tracker
        run_context = None

        if tracker and tracker.is_enabled:
            run_name = f"train-{symbol}-{timeframe}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_context = tracker.start_run(
                run_name=run_name,
                tags={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "command": "train"
                }
            )
            run_context.__enter__()

            # Log configuration parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "n_bars": len(market_data.close),
                "input_dim": input_dim,
                "state_dim": state_dim,
                "multi_timeframe_enabled": additional_timeframes is not None and len(additional_timeframes) > 0,
            }
            if additional_timeframes:
                params["additional_timeframes"] = ",".join(additional_timeframes)
                params["n_additional_timeframes"] = len(additional_timeframes)
            tracker.log_params(params)
            tracker.log_predictor_params(
                self.config.transformer,
                max_seq_length_override=self.config.data.lookback_window
            )
            tracker.log_agent_params(self.config.ppo)

        try:
            # Train predictor
            logger.info("Training prediction model...")
            predictor_results = trainer.train_predictor(
                X_train=splits['train'][0],
                y_train=splits['train'][1],
                X_val=splits['val'][0],
                y_val=splits['val'][1]
            )

            # Train agent
            logger.info("Training RL agent...")
            agent_results = trainer.train_agent(
                env=env,
                total_timesteps=agent_timesteps or self.config.ppo.total_timesteps
            )

            # Save models
            save_dir = os.path.join(self.config.base_dir, self.config.models_dir)
            trainer.save_all(save_dir)

            # Log artifacts to MLflow
            if tracker and tracker.is_enabled:
                config_path = os.path.join(save_dir, 'config.json')
                if os.path.exists(config_path):
                    tracker.log_artifact(config_path)
                history_path = os.path.join(save_dir, 'training_history.json')
                if os.path.exists(history_path):
                    tracker.log_artifact(history_path)

            return {
                'predictor': predictor_results,
                'agent': agent_results
            }

        except Exception:
            # Pass exception info to MLflow for proper run status tracking
            if run_context is not None:
                run_context.__exit__(*sys.exc_info())
                run_context = None  # Prevent double __exit__
            raise

        finally:
            if run_context is not None:
                run_context.__exit__(None, None, None)

    def backtest(
        self,
        market_data,
        _strategy_type: str = 'combined',  # Reserved for future multi-strategy support
        realistic_mode: bool = False,
        enable_monte_carlo: bool = False
    ):
        """Run backtest on historical data.

        Args:
            market_data: Market data to backtest on
            _strategy_type: Strategy type (reserved for future use)
            realistic_mode: If True, applies realistic trading constraints:
                - Minimum 4 hours between trades
                - Maximum 5 trades per day
                - Maximum position size of 10 lots (1M units)
        """
        import pandas as pd

        # Prepare data as DataFrame
        df = pd.DataFrame({
            'open': market_data.open,
            'high': market_data.high,
            'low': market_data.low,
            'close': market_data.close,
            'volume': market_data.volume
        })

        # Add features
        if market_data.features is not None:
            for i, name in enumerate(market_data.feature_names):
                df[name] = market_data.features[:, i]

        # Create backtester
        backtester = Backtester(
            initial_balance=self.config.backtest.initial_balance,
            commission_rate=self.config.backtest.commission_per_lot / 100000,
            spread_pips=self.config.backtest.spread_pips,
            slippage_pips=self.config.backtest.slippage_pips,
            leverage=self.config.backtest.leverage,
            realistic_mode=realistic_mode
        )

        if realistic_mode:
            logger.info("Running backtest with REALISTIC constraints (limited trades, capped position size)")

        # Determine which feature names to use for inference
        # Use saved model feature names if available, otherwise use current features
        if self._model_feature_names:
            inference_feature_names = self._model_feature_names
            # Validate that all required features exist in current data
            current_features = set(market_data.feature_names) if market_data.feature_names else set()
            missing_features = set(inference_feature_names) - current_features
            if missing_features:
                logger.warning(
                    f"Model requires {len(missing_features)} features not in current data: "
                    f"{list(missing_features)[:5]}{'...' if len(missing_features) > 5 else ''}"
                )
        else:
            inference_feature_names = list(market_data.feature_names) if market_data.feature_names else []
            if self._predictor is not None:
                # Check for mismatch when using old model without saved feature names
                current_dim = 5 + len(inference_feature_names)  # OHLCV + features
                if current_dim != self._predictor.input_dim:
                    raise ValueError(
                        f"Feature dimension mismatch: model expects {self._predictor.input_dim} features, "
                        f"but current data has {current_dim} (5 OHLCV + {len(inference_feature_names)} computed). "
                        f"The saved model does not have feature_names metadata, so automatic "
                        f"feature matching is not possible. Please retrain the model with:\n"
                        f"  python main.py train --symbol {market_data.symbol}"
                    )

        # Define strategy
        def strategy(data, predictor=None, agent=None, positions=None):
            """Combined prediction + RL strategy."""
            if len(data) < self.config.data.lookback_window + 1:
                return {'action': 'hold'}

            # Get features for prediction
            if predictor is not None and len(data) >= self.config.data.lookback_window:
                # Prepare input - include OHLCV + computed features to match training
                recent_data = data.tail(self.config.data.lookback_window)
                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                feature_cols = ohlcv_cols + inference_feature_names
                features = recent_data[feature_cols].values

                if features is not None:
                    # Make prediction
                    X = features.reshape(1, self.config.data.lookback_window, -1)
                    prediction = predictor.predict(X)
                    pred_return = prediction['prediction'][0, 0]

                    # Simple threshold-based decision
                    if pred_return > 0.001:
                        return {'action': 'buy', 'stop_loss_pips': 50}
                    elif pred_return < -0.001:
                        return {'action': 'sell', 'stop_loss_pips': 50}

            return {'action': 'hold'}

        # Run backtest
        logger.info("Running backtest...")
        result = backtester.run(
            data=df,
            strategy=strategy,
            predictor=self._predictor
        )

        # Analyze results
        analyzer_config = {
            'enable_monte_carlo': enable_monte_carlo,
            'n_simulations': self.config.backtest.n_simulations,
            'confidence_level': self.config.backtest.confidence_level
        }
        analyzer = PerformanceAnalyzer(config=analyzer_config)
        analysis = analyzer.analyze(result)

        # Print report
        report = analyzer.generate_report(analysis)
        print(report)

        return result, analysis

    def walk_forward_test(self, market_data):
        """Run walk-forward optimization."""
        import pandas as pd

        # Prepare data
        df = pd.DataFrame({
            'open': market_data.open,
            'high': market_data.high,
            'low': market_data.low,
            'close': market_data.close,
            'volume': market_data.volume
        })

        if market_data.features is not None:
            for i, name in enumerate(market_data.feature_names):
                df[name] = market_data.features[:, i]

        # Create walk-forward optimizer
        optimizer = WalkForwardOptimizer(
            train_window=self.config.backtest.train_window_days,
            test_window=self.config.backtest.test_window_days,
            step_size=self.config.backtest.test_window_days
        )

        def train_func(_train_data):
            """Train model on training data."""
            # Simplified training for walk-forward
            return None  # Would train model here

        def backtest_func(test_data, _model):
            """Backtest on test data."""
            backtester = Backtester(
                initial_balance=self.config.backtest.initial_balance
            )

            def simple_strategy(_data, **_kwargs):
                return {'action': 'hold'}

            return backtester.run(test_data, simple_strategy, show_progress=False)

        # Run walk-forward
        results = optimizer.run(df, train_func, backtest_func)

        return results

    def save_models(self, directory: Optional[str] = None):
        """Save all models."""
        if directory is None:
            directory = os.path.join(self.config.base_dir, self.config.models_dir)

        os.makedirs(directory, exist_ok=True)

        # Save model metadata for proper reloading
        metadata = {}

        if self._predictor:
            self._predictor.save(os.path.join(directory, 'predictor.pt'))
            metadata['predictor'] = {
                'input_dim': self._predictor.input_dim,
                'exists': True,
                'feature_names': self._model_feature_names if self._model_feature_names else []
            }

        if self._agent:
            self._agent.save(os.path.join(directory, 'agent.pt'))
            metadata['agent'] = {
                'state_dim': self._agent.state_dim,
                'action_dim': self._agent.action_dim,
                'exists': True
            }

        # Save metadata for model reloading
        metadata_path = os.path.join(directory, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save config
        self.config.save(os.path.join(directory, 'config.json'))

        logger.info(f"Models saved to {directory}")

    def load_models(self, directory: Optional[str] = None):
        """Load all models."""
        if directory is None:
            directory = os.path.join(self.config.base_dir, self.config.models_dir)

        # Track what gets loaded
        loaded_predictor = False
        loaded_agent = False

        # Load config first
        config_path = os.path.join(directory, 'config.json')
        if os.path.exists(config_path):
            self.config = SystemConfig.load(config_path)

        # Load model metadata for dimensions
        metadata_path = os.path.join(directory, 'model_metadata.json')
        if not os.path.exists(metadata_path):
            logger.warning(f"No model metadata found at {metadata_path}")
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load predictor if it exists
        predictor_path = os.path.join(directory, 'predictor.pt')
        predictor_metadata = metadata.get('predictor', {})
        if predictor_metadata.get('exists'):
            if os.path.exists(predictor_path):
                input_dim = predictor_metadata['input_dim']
                # Build config dict matching initialize_models format
                predictor_config = {
                    'd_model': self.config.transformer.d_model,
                    'n_heads': self.config.transformer.n_heads,
                    'n_encoder_layers': self.config.transformer.n_encoder_layers,
                    'd_ff': self.config.transformer.d_ff,
                    'dropout': self.config.transformer.dropout,
                    'max_seq_length': self.config.data.lookback_window,
                    'learning_rate': self.config.transformer.learning_rate,
                    'online_learning_rate': self.config.transformer.online_learning_rate
                }
                self._predictor = TransformerPredictor(
                    input_dim=input_dim,
                    config=predictor_config,
                    device=self.config.device
                )
                self._predictor.load(predictor_path)
                loaded_predictor = True

                # Load feature names used during training
                self._model_feature_names = predictor_metadata.get('feature_names')
                if self._model_feature_names:
                    logger.info(f"Loaded predictor with input_dim={input_dim}, {len(self._model_feature_names)} saved features")
                else:
                    logger.info(f"Loaded predictor with input_dim={input_dim}")
            else:
                logger.warning(f"Predictor metadata exists but file not found: {predictor_path}")

        # Load agent if it exists
        agent_path = os.path.join(directory, 'agent.pt')
        agent_metadata = metadata.get('agent', {})
        if agent_metadata.get('exists'):
            if os.path.exists(agent_path):
                state_dim = agent_metadata['state_dim']
                action_dim = agent_metadata['action_dim']
                # Build config dict matching initialize_models format
                agent_config = {
                    'learning_rate': self.config.ppo.learning_rate,
                    'gamma': self.config.ppo.gamma,
                    'gae_lambda': self.config.ppo.gae_lambda,
                    'clip_epsilon': self.config.ppo.clip_epsilon,
                    'entropy_coef': self.config.ppo.entropy_coef,
                    'n_steps': self.config.ppo.n_steps,
                    'n_epochs': self.config.ppo.n_epochs,
                    'batch_size': self.config.ppo.batch_size,
                    'hidden_sizes': self.config.ppo.actor_hidden_sizes
                }
                self._agent = PPOAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    config=agent_config,
                    device=self.config.device
                )
                self._agent.load(agent_path)
                loaded_agent = True
                logger.info(f"Loaded agent with state_dim={state_dim}, action_dim={action_dim}")
            else:
                logger.warning(f"Agent metadata exists but file not found: {agent_path}")

        # Log summary of what was actually loaded
        if not (loaded_predictor or loaded_agent):
            logger.warning(f"No models loaded from {directory} (check metadata and .pt files)")
        else:
            logger.info(
                f"Models loaded from {directory} "
                f"(predictor={loaded_predictor}, agent={loaded_agent})"
            )


def main():
    """Main entry point."""
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
        help='Trading symbol (default: from config or EURUSD). For multi-symbol training, use --multi-symbol'
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
        '--config', '-c',
        help='Path to config file'
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
        help='MLflow tracking URI (default: mlruns)'
    )

    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking'
    )

    args = parser.parse_args()

    # Load config first (needed for logging setup)
    if args.config and os.path.exists(args.config):
        config = SystemConfig.load(args.config)
    else:
        config = get_config()

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

    # Setup logging with config and CLI overrides
    # CLI args take precedence when explicitly provided (not None)
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
    if args.command == 'train':
        logger.info("Starting training...")

        # Multi-symbol training
        if len(symbols) > 1:
            logger.info(f"Multi-symbol training enabled for: {symbols}")

        all_results = {}
        for symbol in symbols:
            logger.info(f"{'='*50}")
            logger.info(f"Training on {symbol} ({timeframe})")
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

            # Train
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
                system.save_models(symbol_model_dir)
                logger.info(f"Models for {symbol} saved to {symbol_model_dir}")

        # Save to default location for single symbol
        if len(symbols) == 1:
            system.save_models(args.model_dir)

        logger.info("Training complete!")

        # Print summary
        summary = {}
        for symbol, results in all_results.items():
            summary[symbol] = {
                'predictor_final_loss': results['predictor']['train_losses'][-1] if results['predictor']['train_losses'] else None,
                'agent_episodes': len(results['agent']['episode_rewards'])
            }
        print(json.dumps(summary, indent=2))

    elif args.command == 'backtest':
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

    elif args.command == 'walkforward':
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

    elif args.command == 'evaluate':
        logger.info("Evaluating models...")

        market_data = system.load_data(
            symbol=primary_symbol,
            timeframe=timeframe,
            n_bars=n_bars,
            additional_timeframes=additional_timeframes
        )

        if market_data is None:
            sys.exit(1)

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

    elif args.command == 'autotrade':
        if not AUTO_TRADER_AVAILABLE:
            logger.error("Auto-trader not available. MT5 requires Windows.")
            sys.exit(1)

        logger.info("Starting Auto-Trader...")

        # Load models first
        system.load_models(args.model_dir)

        if system._predictor is None or system._agent is None:
            logger.error("Models not loaded. Please train models first.")
            sys.exit(1)

        # Create broker gateway
        broker = MT5BrokerGateway(
            login=config.auto_trader.mt5_login,
            password=config.auto_trader.mt5_password,
            server=config.auto_trader.mt5_server,
            magic_number=config.auto_trader.magic_number
        )

        # Create auto-trader config
        trader_config = AutoTraderConfig(
            symbols=symbols,
            timeframe=timeframe,
            risk_per_trade=config.auto_trader.risk_per_trade,
            max_positions=config.auto_trader.max_positions,
            default_sl_pips=config.auto_trader.default_sl_pips,
            default_tp_pips=config.auto_trader.default_tp_pips,
            paper_mode=args.paper,
            enable_online_learning=config.auto_trader.enable_online_learning
        )

        # Handle online learning manager (may be None if not initialized)
        online_manager = system._online_manager
        if trader_config.enable_online_learning and online_manager is None:
            logger.warning("Online learning enabled but OnlineLearningManager not initialized. "
                         "Online learning will be disabled.")
            trader_config.enable_online_learning = False

        # Create auto-trader
        auto_trader = AutoTrader(
            broker=broker,
            predictor=system._predictor,
            agent=system._agent,
            risk_manager=system.risk_manager,
            online_manager=online_manager,
            data_pipeline=system.data_pipeline,
            config=trader_config
        )

        # Start trading
        mode = "PAPER" if args.paper else "LIVE"
        print(f"\n{'='*60}")
        print(f"  LEAP AUTO-TRADER - {mode} MODE")
        print(f"  Symbols: {', '.join(symbols)}")
        print(f"  Risk per trade: {trader_config.risk_per_trade*100:.1f}%")
        print(f"  Max positions: {trader_config.max_positions}")
        print(f"{'='*60}\n")

        if not args.paper:
            print("WARNING: LIVE TRADING MODE - Real money at risk!")
            confirm = input("Type 'CONFIRM' to proceed: ")
            if confirm != 'CONFIRM':
                print("Aborted.")
                sys.exit(0)

        try:
            auto_trader.start()
            print("Auto-trader running. Press Ctrl+C to stop.")

            # Keep main thread alive
            while auto_trader.state.value == 'running':
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nStopping auto-trader...")
            auto_trader.stop()

        # Print final statistics
        stats = auto_trader.get_statistics()
        print("\n" + "="*60)
        print("  SESSION SUMMARY")
        print("="*60)
        if stats.get('session'):
            session = stats['session']
            print(f"  Duration: {session.get('duration', 'N/A')}")
            print(f"  Total Trades: {session.get('total_trades', 0)}")
            print(f"  Win Rate: {session.get('win_rate', 0)*100:.1f}%")
            print(f"  P&L: ${session.get('pnl', 0):.2f} ({session.get('pnl_percent', 0):.2f}%)")
            print(f"  Max Drawdown: {session.get('max_drawdown', 0)*100:.1f}%")
        print("="*60 + "\n")


if __name__ == '__main__':
    main()
