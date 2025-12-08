"""
Leap Trading System - Main Orchestrator
Command-line interface and main entry point for the trading system.
"""

import argparse
import logging
import sys
import os
import json
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

import numpy as np


# Configure logging
def setup_logging(level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
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

        logger.info("Leap Trading System initialized")

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
        n_bars: int = 50000
    ):
        """Load and prepare market data."""
        logger.info(f"Loading data for {symbol} {timeframe}...")

        # Connect to data source
        self.data_pipeline.connect()

        # Fetch historical data
        market_data = self.data_pipeline.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            n_bars=n_bars
        )

        if market_data is None:
            logger.error("Failed to load market data")
            return None

        logger.info(f"Loaded {len(market_data.close)} bars with {len(market_data.feature_names)} features")

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
        agent_timesteps: Optional[int] = None
    ):
        """Train both models."""
        # Prepare data
        splits, input_dim = self.prepare_training_data(market_data)

        # Create environment
        env = self.create_environment(market_data)
        state_dim = env.observation_space.shape[0]

        # Initialize models
        self.initialize_models(input_dim, state_dim)

        # Create trainer
        trainer = ModelTrainer(
            predictor=self._predictor,
            agent=self._agent,
            data_pipeline=self.data_pipeline,
            config={
                'predictor_epochs': predictor_epochs or self.config.transformer.epochs,
                'agent_timesteps': agent_timesteps or self.config.ppo.total_timesteps,
                'batch_size': self.config.transformer.batch_size,
                'checkpoint_dir': os.path.join(self.config.base_dir, self.config.checkpoints_dir)
            }
        )

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

        return {
            'predictor': predictor_results,
            'agent': agent_results
        }

    def backtest(
        self,
        market_data,
        _strategy_type: str = 'combined'  # Reserved for future multi-strategy support
    ):
        """Run backtest on historical data."""
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
            leverage=self.config.backtest.leverage
        )

        # Define strategy
        def strategy(data, predictor=None, agent=None, positions=None):
            """Combined prediction + RL strategy."""
            if len(data) < self.config.data.lookback_window + 1:
                return {'action': 'hold'}

            # Get features for prediction
            if predictor is not None and len(data) >= self.config.data.lookback_window:
                # Prepare input
                recent_data = data.tail(self.config.data.lookback_window)
                features = recent_data[market_data.feature_names].values if market_data.feature_names else None

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
        analyzer = PerformanceAnalyzer()
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

            return backtester.run(test_data, simple_strategy)

        # Run walk-forward
        results = optimizer.run(df, train_func, backtest_func)

        return results

    def start_live_trading(self, paper: bool = True):
        """Start live trading (paper or real)."""
        mode = 'paper' if paper else 'live'
        logger.info(f"Starting {mode} trading...")

        # Ensure models are loaded
        if self._predictor is None or self._agent is None:
            logger.error("Models not loaded. Please train or load models first.")
            return

        # Create online learning manager
        self._online_manager = OnlineLearningManager(
            predictor=self._predictor,
            agent=self._agent
        )

        # Main trading loop would go here
        logger.info(f"{mode.capitalize()} trading started. Press Ctrl+C to stop.")

        try:
            while True:
                # Get latest market data
                # Make prediction
                # Execute trades
                # Online learning updates
                import time
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Trading stopped by user")

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
                'exists': True
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
        if metadata.get('predictor', {}).get('exists') and os.path.exists(predictor_path):
            input_dim = metadata['predictor']['input_dim']
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
            logger.info(f"Loaded predictor with input_dim={input_dim}")

        # Load agent if it exists
        agent_path = os.path.join(directory, 'agent.pt')
        if metadata.get('agent', {}).get('exists') and os.path.exists(agent_path):
            state_dim = metadata['agent']['state_dim']
            action_dim = metadata['agent']['action_dim']
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
            logger.info(f"Loaded agent with state_dim={state_dim}, action_dim={action_dim}")

        logger.info(f"Models loaded from {directory}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Leap Trading System - AI-Powered Forex Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --symbol EURUSD --epochs 100
  python main.py backtest --symbol EURUSD
  python main.py live --paper
  python main.py evaluate --model-dir ./models
        """
    )

    parser.add_argument(
        'command',
        choices=['train', 'backtest', 'live', 'evaluate', 'walkforward'],
        help='Command to execute'
    )

    parser.add_argument(
        '--symbol', '-s',
        default='EURUSD',
        help='Trading symbol (default: EURUSD)'
    )

    parser.add_argument(
        '--timeframe', '-t',
        default='1h',
        help='Timeframe (default: 1h)'
    )

    parser.add_argument(
        '--bars', '-b',
        type=int,
        default=50000,
        help='Number of bars to load (default: 50000)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Training epochs for predictor (default: 100)'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Training timesteps for agent (default: 100000)'
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
        '--config', '-c',
        help='Path to config file'
    )

    parser.add_argument(
        '--log-level', '-l',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    parser.add_argument(
        '--log-file',
        help='Log file path'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Load config
    if args.config and os.path.exists(args.config):
        config = SystemConfig.load(args.config)
    else:
        config = get_config()

    # Create system
    system = LeapTradingSystem(config)

    # Execute command
    if args.command == 'train':
        logger.info("Starting training...")

        # Load data
        market_data = system.load_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            n_bars=args.bars
        )

        if market_data is None:
            logger.error("Failed to load data. Exiting.")
            sys.exit(1)

        # Train
        results = system.train(
            market_data=market_data,
            predictor_epochs=args.epochs,
            agent_timesteps=args.timesteps
        )

        # Save
        system.save_models(args.model_dir)

        logger.info("Training complete!")
        print(json.dumps({
            'predictor_final_loss': results['predictor']['train_losses'][-1] if results['predictor']['train_losses'] else None,
            'agent_episodes': len(results['agent']['episode_rewards'])
        }, indent=2))

    elif args.command == 'backtest':
        logger.info("Starting backtest...")

        # Load data
        market_data = system.load_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            n_bars=args.bars
        )

        if market_data is None:
            sys.exit(1)

        # Load models if available
        if os.path.exists(args.model_dir):
            system.load_models(args.model_dir)

        # Run backtest
        _result, analysis = system.backtest(market_data)

        # Save results
        results_dir = os.path.join(config.base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(results_dir, f'backtest_{timestamp}.json'), 'w') as f:
            json.dump({
                'total_return': analysis.get('total_return'),
                'sharpe_ratio': analysis.get('sharpe_ratio'),
                'max_drawdown': analysis.get('max_drawdown'),
                'win_rate': analysis.get('win_rate'),
                'total_trades': analysis.get('total_trades')
            }, f, indent=2)

    elif args.command == 'walkforward':
        logger.info("Starting walk-forward analysis...")

        market_data = system.load_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            n_bars=args.bars
        )

        if market_data is None:
            sys.exit(1)

        results = system.walk_forward_test(market_data)
        print(json.dumps(results, indent=2, default=str))

    elif args.command == 'live':
        system.load_models(args.model_dir)
        system.start_live_trading(paper=args.paper)

    elif args.command == 'evaluate':
        logger.info("Evaluating models...")

        market_data = system.load_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            n_bars=args.bars
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


if __name__ == '__main__':
    main()
