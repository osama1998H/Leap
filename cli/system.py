"""
LeapTradingSystem - Main orchestrator for the Leap Trading System.

This module contains the core LeapTradingSystem class that coordinates
all trading system components including data pipeline, models, risk management,
backtesting, and online learning.
"""

import logging
import os
import sys
import json
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import numpy as np

from config import SystemConfig, get_config
from core.data_pipeline import DataPipeline
from core.trading_env import TradingEnvironment
from core.trading_types import Action
from core.risk_manager import RiskManager, DynamicRiskManager
from models import (
    create_predictor,
    create_agent,
    PredictorModel,
    AgentModel,
)
from training.trainer import ModelTrainer
from evaluation.backtester import Backtester, WalkForwardOptimizer
from evaluation.metrics import PerformanceAnalyzer
from utils.mlflow_tracker import MLflowTracker, create_tracker
from core.strategy import CombinedPredictorAgentStrategy, StrategyConfig

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

        # Environment dimensions from training (for live trading compatibility)
        self._model_env_config = {
            'window_size': self.config.data.lookback_window,
            'n_additional_features': 0,
            'n_account_features': 8
        }

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
    def predictor(self) -> Optional[PredictorModel]:
        """Get predictor (initialized via initialize_models or load_models)."""
        return self._predictor

    @property
    def agent(self) -> Optional[AgentModel]:
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
        """Initialize prediction and RL models using factory functions."""
        logger.info(f"Initializing models (input_dim={input_dim}, state_dim={state_dim})...")

        # Initialize predictor via factory
        self._predictor = create_predictor(
            model_type='transformer',
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

        # Initialize agent via factory
        self._agent = create_agent(
            model_type='ppo',
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

        # Create environment(s) - split data for eval if patience is enabled
        eval_env = None
        eval_split = getattr(self.config.ppo, 'eval_split', 0.2)

        if self.config.ppo.patience > 0 and eval_split > 0 and hasattr(market_data, 'iloc'):
            # Split market_data for train/eval to enable early stopping
            n_bars = len(market_data.close)
            split_idx = int(n_bars * (1 - eval_split))
            eval_bars = n_bars - split_idx
            window_size = self.config.data.lookback_window

            # Only split if eval portion is large enough for the window size
            if eval_bars > window_size + 10:  # Need at least window_size + some steps
                train_data = market_data.iloc(slice(0, split_idx))
                eval_data = market_data.iloc(slice(split_idx, None))

                env = self.create_environment(train_data)
                eval_env = self.create_environment(eval_data)
                logger.info(f"Created train env ({split_idx} bars) and eval env ({eval_bars} bars) for early stopping")
            else:
                logger.warning(f"Eval split would create too small eval set ({eval_bars} bars < {window_size} window). Disabling early stopping.")
                env = self.create_environment(market_data)
        else:
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
                'ppo_patience': self.config.ppo.patience,
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
                eval_env=eval_env,
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

    def train_predictor_only(
        self,
        market_data,
        predictor_epochs: Optional[int] = None,
        symbol: str = 'EURUSD',
        timeframe: str = '1h',
        additional_timeframes: Optional[list] = None
    ):
        """Train only the Transformer predictor model.

        Args:
            market_data: Market data for training
            predictor_epochs: Number of epochs for predictor training
            symbol: Trading symbol (for MLflow tracking)
            timeframe: Timeframe (for MLflow tracking)
            additional_timeframes: List of additional timeframes used for features
        """
        # Store feature names used during training for inference compatibility
        if market_data.feature_names:
            self._model_feature_names = list(market_data.feature_names)
            logger.info(f"Training predictor with {len(self._model_feature_names)} computed features")

        # Prepare data
        splits, input_dim = self.prepare_training_data(market_data)

        # Create environment to get state_dim (needed for metadata compatibility)
        env = self.create_environment(market_data)
        state_dim = env.observation_space.shape[0]

        # Initialize models (predictor will be trained, agent initialized for metadata)
        self.initialize_models(input_dim, state_dim)

        # Create trainer with MLflow tracker
        trainer = ModelTrainer(
            predictor=self._predictor,
            agent=self._agent,
            data_pipeline=self.data_pipeline,
            config={
                'predictor_epochs': predictor_epochs or self.config.transformer.epochs,
                'agent_timesteps': self.config.ppo.total_timesteps,
                'batch_size': self.config.transformer.batch_size,
                'patience': self.config.transformer.patience,
                'ppo_patience': self.config.ppo.patience,
                'checkpoint_dir': os.path.join(self.config.base_dir, self.config.checkpoints_dir)
            },
            mlflow_tracker=self.mlflow_tracker
        )

        # Start MLflow run if enabled
        tracker = self.mlflow_tracker
        run_context = None

        if tracker and tracker.is_enabled:
            run_name = f"train-predictor-{symbol}-{timeframe}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_context = tracker.start_run(
                run_name=run_name,
                tags={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "command": "train",
                    "model_type": "transformer"
                }
            )
            run_context.__enter__()

            # Log configuration parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "n_bars": len(market_data.close),
                "input_dim": input_dim,
                "model_type": "transformer",
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

        try:
            # Train predictor only
            logger.info("Training prediction model...")
            predictor_results = trainer.train_predictor(
                X_train=splits['train'][0],
                y_train=splits['train'][1],
                X_val=splits['val'][0],
                y_val=splits['val'][1]
            )

            # Save predictor only (preserve existing agent metadata if present)
            save_dir = os.path.join(self.config.base_dir, self.config.models_dir)
            self._save_predictor_only(save_dir)

            # Log artifacts to MLflow
            if tracker and tracker.is_enabled:
                config_path = os.path.join(save_dir, 'config.json')
                if os.path.exists(config_path):
                    tracker.log_artifact(config_path)

            return {
                'predictor': predictor_results,
                'agent': None
            }

        except Exception:
            if run_context is not None:
                run_context.__exit__(*sys.exc_info())
                run_context = None
            raise

        finally:
            if run_context is not None:
                run_context.__exit__(None, None, None)

    def train_agent_only(
        self,
        market_data,
        agent_timesteps: Optional[int] = None,
        symbol: str = 'EURUSD',
        timeframe: str = '1h',
        additional_timeframes: Optional[list] = None
    ):
        """Train only the PPO agent model.

        Args:
            market_data: Market data for training
            agent_timesteps: Number of timesteps for agent training
            symbol: Trading symbol (for MLflow tracking)
            timeframe: Timeframe (for MLflow tracking)
            additional_timeframes: List of additional timeframes used for features
        """
        # Store feature names used during training for inference compatibility
        if market_data.feature_names:
            self._model_feature_names = list(market_data.feature_names)
            logger.info(f"Training agent with {len(self._model_feature_names)} computed features")

        # Prepare data (needed for input_dim even if not training predictor)
        splits, input_dim = self.prepare_training_data(market_data)

        # Create environment(s) - split data for eval if patience is enabled
        eval_env = None
        eval_split = getattr(self.config.ppo, 'eval_split', 0.2)

        if self.config.ppo.patience > 0 and eval_split > 0 and hasattr(market_data, 'iloc'):
            # Split market_data for train/eval to enable early stopping
            n_bars = len(market_data.close)
            split_idx = int(n_bars * (1 - eval_split))
            eval_bars = n_bars - split_idx
            window_size = self.config.data.lookback_window

            # Only split if eval portion is large enough for the window size
            if eval_bars > window_size + 10:  # Need at least window_size + some steps
                train_data = market_data.iloc(slice(0, split_idx))
                eval_data = market_data.iloc(slice(split_idx, None))

                env = self.create_environment(train_data)
                eval_env = self.create_environment(eval_data)
                logger.info(f"Created train env ({split_idx} bars) and eval env ({eval_bars} bars) for early stopping")
            else:
                logger.warning(f"Eval split would create too small eval set ({eval_bars} bars < {window_size} window). Disabling early stopping.")
                env = self.create_environment(market_data)
        else:
            env = self.create_environment(market_data)

        state_dim = env.observation_space.shape[0]

        # Initialize models (agent will be trained, predictor initialized for metadata)
        self.initialize_models(input_dim, state_dim)

        # Create trainer with MLflow tracker
        trainer = ModelTrainer(
            predictor=self._predictor,
            agent=self._agent,
            data_pipeline=self.data_pipeline,
            config={
                'predictor_epochs': self.config.transformer.epochs,
                'agent_timesteps': agent_timesteps or self.config.ppo.total_timesteps,
                'batch_size': self.config.transformer.batch_size,
                'patience': self.config.transformer.patience,
                'ppo_patience': self.config.ppo.patience,
                'checkpoint_dir': os.path.join(self.config.base_dir, self.config.checkpoints_dir)
            },
            mlflow_tracker=self.mlflow_tracker
        )

        # Start MLflow run if enabled
        tracker = self.mlflow_tracker
        run_context = None

        if tracker and tracker.is_enabled:
            run_name = f"train-agent-{symbol}-{timeframe}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_context = tracker.start_run(
                run_name=run_name,
                tags={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "command": "train",
                    "model_type": "ppo"
                }
            )
            run_context.__enter__()

            # Log configuration parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "n_bars": len(market_data.close),
                "state_dim": state_dim,
                "model_type": "ppo",
                "multi_timeframe_enabled": additional_timeframes is not None and len(additional_timeframes) > 0,
            }
            if additional_timeframes:
                params["additional_timeframes"] = ",".join(additional_timeframes)
                params["n_additional_timeframes"] = len(additional_timeframes)
            tracker.log_params(params)
            tracker.log_agent_params(self.config.ppo)

        try:
            # Train agent only
            logger.info("Training RL agent...")
            agent_results = trainer.train_agent(
                env=env,
                eval_env=eval_env,
                total_timesteps=agent_timesteps or self.config.ppo.total_timesteps
            )

            # Save agent only (preserve existing predictor metadata if present)
            save_dir = os.path.join(self.config.base_dir, self.config.models_dir)
            self._save_agent_only(save_dir)

            # Log artifacts to MLflow
            if tracker and tracker.is_enabled:
                config_path = os.path.join(save_dir, 'config.json')
                if os.path.exists(config_path):
                    tracker.log_artifact(config_path)

            return {
                'predictor': None,
                'agent': agent_results
            }

        except Exception:
            if run_context is not None:
                run_context.__exit__(*sys.exc_info())
                run_context = None
            raise

        finally:
            if run_context is not None:
                run_context.__exit__(None, None, None)

    def _save_predictor_only(self, directory: str):
        """Save only the predictor model and update metadata.

        Preserves existing agent metadata if present.
        """
        os.makedirs(directory, exist_ok=True)

        # Load existing metadata if present
        metadata_path = os.path.join(directory, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Update predictor metadata
        if self._predictor:
            self._predictor.save(os.path.join(directory, 'predictor.pt'))
            metadata['predictor'] = {
                'input_dim': self._predictor.input_dim,
                'exists': True,
                'feature_names': self._model_feature_names or []
            }

        # Update environment metadata
        n_features = len(self._model_feature_names) if self._model_feature_names else 0
        metadata['environment'] = {
            'window_size': self.config.data.lookback_window,
            'n_additional_features': n_features,
            'n_account_features': 8
        }

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.config.save(os.path.join(directory, 'config.json'))
        logger.info(f"Predictor saved to {directory}")

    def _save_agent_only(self, directory: str):
        """Save only the agent model and update metadata.

        Preserves existing predictor metadata if present.
        """
        os.makedirs(directory, exist_ok=True)

        # Load existing metadata if present
        metadata_path = os.path.join(directory, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Update agent metadata
        if self._agent:
            self._agent.save(os.path.join(directory, 'agent.pt'))
            metadata['agent'] = {
                'state_dim': self._agent.state_dim,
                'action_dim': self._agent.action_dim,
                'exists': True
            }

        # Update environment metadata if not present
        if 'environment' not in metadata:
            n_features = len(self._model_feature_names) if self._model_feature_names else 0
            metadata['environment'] = {
                'window_size': self.config.data.lookback_window,
                'n_additional_features': n_features,
                'n_account_features': 8
            }

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.config.save(os.path.join(directory, 'config.json'))
        logger.info(f"Agent saved to {directory}")

    def backtest(
        self,
        market_data,
        realistic_mode: bool = False,
        enable_monte_carlo: bool = False
    ):
        """Run backtest on historical data.

        Args:
            market_data: Market data to backtest on
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

        # Add features - use pd.concat to avoid DataFrame fragmentation
        if market_data.features is not None:
            feature_dict = {
                name: market_data.features[:, i]
                for i, name in enumerate(market_data.feature_names)
            }
            feature_df = pd.DataFrame(feature_dict)
            df = pd.concat([df, feature_df], axis=1)

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

        # Configuration for signal combination (from AutoTrader config for consistency)
        prediction_threshold = getattr(self.config.auto_trader, 'prediction_threshold', 0.001)
        min_confidence = getattr(self.config.auto_trader, 'min_confidence', 0.6)
        sl_pips = getattr(self.config.auto_trader, 'default_sl_pips', 50)
        tp_pips = getattr(self.config.auto_trader, 'default_tp_pips', 100)

        # Create strategy instance using TradingStrategy pattern
        # This replaces the previous inline helper functions and strategy callable
        strategy_config = StrategyConfig(
            min_confidence=min_confidence,
            prediction_threshold=prediction_threshold,
            default_sl_pips=sl_pips,
            default_tp_pips=tp_pips,
            risk_per_trade=0.02,
            lookback_window=self.config.data.lookback_window
        )

        # Use CombinedPredictorAgentStrategy - single source of truth for signal logic
        strategy = CombinedPredictorAgentStrategy(
            predictor=self._predictor,
            agent=self._agent,
            config=strategy_config,
            feature_names=inference_feature_names
        )

        # Run backtest with strategy instance
        logger.info("Running backtest...")
        if self._agent is not None:
            logger.info(f"Using strategy: {strategy.name} (Transformer + PPO)")
        else:
            logger.info(f"Using strategy: {strategy.name} (Transformer only)")

        result = backtester.run(
            data=df,
            strategy=strategy,
            predictor=self._predictor,
            agent=self._agent
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
        """Run walk-forward optimization.

        This method performs proper walk-forward validation:
        1. Split data into train/test folds
        2. Train a FRESH model on each fold's training data
        3. Backtest the trained model on unseen test data
        4. Aggregate results across all folds

        This validates if the strategy works across different time periods.
        """
        import pandas as pd
        from numpy.lib.stride_tricks import sliding_window_view

        # Prepare data
        df = pd.DataFrame({
            'open': market_data.open,
            'high': market_data.high,
            'low': market_data.low,
            'close': market_data.close,
            'volume': market_data.volume
        })

        # Store feature names for use in strategy
        feature_names = list(market_data.feature_names) if market_data.feature_names else []

        if market_data.features is not None:
            for i, name in enumerate(feature_names):
                df[name] = market_data.features[:, i]

        # Create walk-forward optimizer
        optimizer = WalkForwardOptimizer(
            train_window=self.config.backtest.train_window_days,
            test_window=self.config.backtest.test_window_days,
            step_size=self.config.backtest.test_window_days
        )

        # Configuration for walk-forward training
        lookback_window = self.config.data.lookback_window
        prediction_horizon = self.config.data.prediction_horizon
        wf_epochs = self.config.backtest.walk_forward_epochs
        device = self.config.device

        def train_func(train_data):
            """Train a fresh predictor model on the fold's training data."""
            try:
                # Prepare training sequences from the fold's data
                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                all_cols = ohlcv_cols + feature_names

                # Ensure we have all columns
                available_cols = [c for c in all_cols if c in train_data.columns]
                train_features = train_data[available_cols].values

                # Create sequences (X) and targets (y) - VECTORIZED
                n_samples = len(train_features) - lookback_window - prediction_horizon

                if n_samples < 100:
                    logger.warning(f"Insufficient training data in fold: {n_samples} sequences")
                    return None

                # Create sliding windows: shape (n_windows, lookback_window, n_features)
                X_train = sliding_window_view(train_features, window_shape=lookback_window, axis=0)
                X_train = X_train[:n_samples].copy()  # Copy for contiguous array

                # Vectorized target computation: returns from current to future close
                close_prices = train_data['close'].values
                current_indices = np.arange(lookback_window - 1, lookback_window - 1 + n_samples)
                future_indices = current_indices + prediction_horizon

                current_close = close_prices[current_indices]
                future_close = close_prices[future_indices]
                targets = (future_close - current_close) / (current_close + 1e-10)

                y_train = targets.reshape(-1, 1)

                # Split into train/val (80/20)
                split_idx = int(len(X_train) * 0.8)
                X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
                y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

                input_dim = X_train.shape[2]

                # Create fresh predictor for this fold via factory
                predictor = create_predictor(
                    model_type='transformer',
                    input_dim=input_dim,
                    config={
                        'd_model': self.config.transformer.d_model,
                        'n_heads': self.config.transformer.n_heads,
                        'n_encoder_layers': self.config.transformer.n_encoder_layers,
                        'd_ff': self.config.transformer.d_ff,
                        'dropout': self.config.transformer.dropout,
                        'max_seq_length': lookback_window,
                        'learning_rate': self.config.transformer.learning_rate,
                        'weight_decay': self.config.transformer.weight_decay
                    },
                    device=device
                )

                # Train with reduced epochs for walk-forward
                predictor.train(
                    X_train=X_tr,
                    y_train=y_tr,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=wf_epochs,
                    batch_size=self.config.transformer.batch_size,
                    patience=max(5, wf_epochs // 4),  # Reduced patience
                    verbose=False
                )

                return {
                    'predictor': predictor,
                    'feature_names': feature_names,
                    'input_dim': input_dim
                }

            except Exception as e:
                logger.warning(f"Training failed in fold: {e}")
                return None

        def backtest_func(test_data, model):
            """Backtest using the trained predictor on test data."""
            bt = Backtester(
                initial_balance=self.config.backtest.initial_balance,
                commission_rate=self.config.backtest.commission_per_lot / 100000,
                slippage_pips=self.config.backtest.slippage_pips,
                spread_pips=self.config.backtest.spread_pips,
                leverage=self.config.backtest.leverage
            )

            # If no model was trained, use a simple hold strategy
            if model is None:
                def hold_strategy(_data, **_kwargs):
                    return {'action': 'hold'}
                return bt.run(test_data, hold_strategy, show_progress=False)

            predictor = model['predictor']
            model_feature_names = model['feature_names']
            expected_input_dim = model['input_dim']

            def wf_strategy(data, **kwargs):
                """Strategy using the fold's trained predictor."""
                if len(data) < lookback_window + 1:
                    return {'action': 'hold'}

                try:
                    # Prepare input features
                    recent_data = data.tail(lookback_window)
                    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                    feature_cols = ohlcv_cols + model_feature_names

                    # Only use columns that exist
                    available_cols = [c for c in feature_cols if c in recent_data.columns]

                    # Validate feature dimensions match training
                    if len(available_cols) != expected_input_dim:
                        logger.debug(
                            f"Feature mismatch: expected {expected_input_dim}, got {len(available_cols)}"
                        )
                        return {'action': 'hold'}

                    features = recent_data[available_cols].values

                    if features is None or len(features) < lookback_window:
                        return {'action': 'hold'}

                    # Make prediction
                    X = features.reshape(1, lookback_window, -1)
                    prediction = predictor.predict(X)
                    pred_return = prediction['prediction'][0, 0]

                    # Trading decision based on predicted return
                    if pred_return > 0.001:  # Predict +0.1% return
                        return {'action': 'buy', 'stop_loss_pips': 50, 'take_profit_pips': 100}
                    elif pred_return < -0.001:  # Predict -0.1% return
                        return {'action': 'sell', 'stop_loss_pips': 50, 'take_profit_pips': 100}

                except Exception as e:
                    logger.debug(f"Prediction failed in wf_strategy: {e}")

                return {'action': 'hold'}

            return bt.run(test_data, wf_strategy, show_progress=False)

        # Run walk-forward with parallel disabled by default (GPU contention)
        logger.info("Starting walk-forward optimization with model training per fold...")
        logger.info(f"  Training epochs per fold: {wf_epochs}")
        logger.info(f"  Train window: {self.config.backtest.train_window_days} days")
        logger.info(f"  Test window: {self.config.backtest.test_window_days} days")

        results = optimizer.run(
            df,
            train_func,
            backtest_func,
            parallel=self.config.backtest.walk_forward_parallel
        )

        return results

    def save_models(self, directory: Optional[str] = None):
        """Save all models."""
        if directory is None:
            directory = os.path.join(self.config.base_dir, self.config.models_dir)

        os.makedirs(directory, exist_ok=True)

        # Save model metadata for proper reloading
        metadata = {}

        # Store training environment dimensions for live trading compatibility
        n_additional_features = len(self._model_feature_names) if self._model_feature_names else 0
        metadata['environment'] = {
            'window_size': self.config.data.lookback_window,
            'n_additional_features': n_additional_features,
            'n_account_features': 8  # Training env uses 8 account features
        }

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

        # Load environment configuration for live trading compatibility
        env_metadata = metadata.get('environment', {})
        if env_metadata:
            self._model_env_config = {
                'window_size': env_metadata.get('window_size', self.config.data.lookback_window),
                'n_additional_features': env_metadata.get('n_additional_features', 0),
                'n_account_features': env_metadata.get('n_account_features', 8)
            }
            logger.info(f"Loaded environment config: window_size={self._model_env_config['window_size']}, "
                       f"n_additional_features={self._model_env_config['n_additional_features']}")
        else:
            # Backward compatibility: infer from predictor metadata if available
            predictor_meta = metadata.get('predictor', {})
            feature_names = predictor_meta.get('feature_names', [])
            if feature_names:
                self._model_env_config['n_additional_features'] = len(feature_names)
                logger.info(f"Inferred n_additional_features={len(feature_names)} from feature_names (backward compat)")

        # Load predictor if it exists
        predictor_path = os.path.join(directory, 'predictor.pt')
        predictor_metadata = metadata.get('predictor', {})
        if predictor_metadata.get('exists'):
            if os.path.exists(predictor_path):
                input_dim = predictor_metadata['input_dim']
                # Use loaded window_size from model metadata for compatibility
                window_size = self._model_env_config.get('window_size', self.config.data.lookback_window)
                # Build config dict matching initialize_models format
                predictor_config = {
                    'd_model': self.config.transformer.d_model,
                    'n_heads': self.config.transformer.n_heads,
                    'n_encoder_layers': self.config.transformer.n_encoder_layers,
                    'd_ff': self.config.transformer.d_ff,
                    'dropout': self.config.transformer.dropout,
                    'max_seq_length': window_size,
                    'learning_rate': self.config.transformer.learning_rate,
                    'online_learning_rate': self.config.transformer.online_learning_rate
                }
                # Create predictor via factory (defaults to 'transformer', future: read from checkpoint)
                self._predictor = create_predictor(
                    model_type='transformer',
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
                # Create agent via factory (defaults to 'ppo', future: read from checkpoint)
                self._agent = create_agent(
                    model_type='ppo',
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
