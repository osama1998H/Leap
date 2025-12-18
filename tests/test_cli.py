"""
Leap Trading System - CLI Tests
Comprehensive tests for the command-line interface and LeapTradingSystem.
"""

import pytest
import numpy as np
import os
import sys
import json
import tempfile
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from io import StringIO
import argparse

# Import components to test
from main import LeapTradingSystem, main, initialize_logging
from config import (
    SystemConfig,
    get_config,
    load_training_config,
    load_data_config,
    load_backtest_config,
    load_risk_config,
    load_auto_trader_config,
    load_logging_config,
)


# ============================================================================
# Mock Classes
# ============================================================================

class MockMarketData:
    """Mock market data for testing."""

    def __init__(self, n_bars=1000, symbol='EURUSD'):
        self.symbol = symbol
        self.close = np.random.uniform(1.0, 1.5, n_bars)
        self.open = self.close * (1 + np.random.uniform(-0.001, 0.001, n_bars))
        self.high = np.maximum(self.close, self.open) * (1 + np.random.uniform(0, 0.002, n_bars))
        self.low = np.minimum(self.close, self.open) * (1 - np.random.uniform(0, 0.002, n_bars))
        self.volume = np.random.uniform(1000, 10000, n_bars)
        self.features = np.random.randn(n_bars, 50).astype(np.float32)
        self.feature_names = [f'feature_{i}' for i in range(50)]


class MockDataPipeline:
    """Mock DataPipeline for testing."""

    def __init__(self, config=None):
        self.connected = False
        self._config = config

    def connect(self):
        self.connected = True

    def fetch_historical_data(self, symbol, timeframe, n_bars):
        return MockMarketData(n_bars=n_bars, symbol=symbol)

    def prepare_sequences(self, data, sequence_length=60, prediction_horizon=12):
        n_samples = max(100, len(data.close) - sequence_length - prediction_horizon)
        n_features = 55  # OHLCV + computed features
        X = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
        y = np.random.randn(n_samples).astype(np.float32)
        timestamps = [None] * n_samples
        return X, y, timestamps

    def create_train_val_test_split(self, X, y, train_ratio=0.7, val_ratio=0.15):
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return {
            'train': (X[:train_end], y[:train_end]),
            'val': (X[train_end:val_end], y[train_end:val_end]),
            'test': (X[val_end:], y[val_end:])
        }


class MockPredictor:
    """Mock TransformerPredictor."""

    def __init__(self, input_dim=55, config=None, device='cpu'):
        self.input_dim = input_dim
        self.model = Mock()

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32,
              verbose=True, mlflow_callback=None):
        return {
            'train_losses': [0.5 - i * 0.01 for i in range(epochs)],
            'val_losses': [0.6 - i * 0.01 for i in range(epochs)],
            'best_val_loss': 0.4
        }

    def predict(self, X, return_uncertainty=False):
        batch_size = X.shape[0]
        return {
            'prediction': np.random.randn(batch_size, 1),
            'quantiles': np.random.randn(batch_size, 3),
            'uncertainty': np.abs(np.random.randn(batch_size))
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('mock_predictor')

    def load(self, path):
        pass


class MockAgent:
    """Mock PPOAgent."""

    def __init__(self, state_dim=100, action_dim=4, config=None, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = Mock()

    def train_on_env(self, env, total_timesteps=1000, eval_env=None,
                     eval_frequency=100, mlflow_callback=None):
        n_episodes = total_timesteps // 100
        return {
            'episode_rewards': [np.random.uniform(-100, 200) for _ in range(n_episodes)],
            'episode_lengths': [np.random.randint(50, 150) for _ in range(n_episodes)]
        }

    def select_action(self, state):
        return 0, -0.5, 0.1

    def evaluate(self, env, n_episodes=10):
        return 50.0

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('mock_agent')

    def load(self, path):
        pass


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config(temp_dir):
    """Create mock configuration."""
    config = get_config()
    config.base_dir = temp_dir
    config.models_dir = 'models'
    config.logs_dir = 'logs'
    config.checkpoints_dir = 'checkpoints'
    config.mlflow.enabled = False
    return config


@pytest.fixture
def trading_system(mock_config):
    """Create LeapTradingSystem with mock config."""
    return LeapTradingSystem(config=mock_config)


# ============================================================================
# LeapTradingSystem Initialization Tests
# ============================================================================

class TestLeapTradingSystemInitialization:
    """Tests for LeapTradingSystem initialization."""

    def test_initialization_default_config(self, temp_dir):
        """Test initialization with default config."""
        with patch('cli.parser.get_config') as mock_get_config:
            config = get_config()
            config.base_dir = temp_dir
            config.mlflow.enabled = False
            mock_get_config.return_value = config

            system = LeapTradingSystem()

            assert system.config is not None
            assert system._data_pipeline is None  # Lazy loading
            assert system._predictor is None
            assert system._agent is None

    def test_initialization_custom_config(self, mock_config):
        """Test initialization with custom config."""
        system = LeapTradingSystem(config=mock_config)

        assert system.config == mock_config

    def test_initialization_creates_directories(self, mock_config):
        """Test that initialization creates necessary directories."""
        _system = LeapTradingSystem(config=mock_config)  # noqa: F841

        models_dir = os.path.join(mock_config.base_dir, 'models')
        logs_dir = os.path.join(mock_config.base_dir, 'logs')

        assert os.path.exists(models_dir)
        assert os.path.exists(logs_dir)

    def test_lazy_loading_data_pipeline(self, trading_system):
        """Test lazy loading of data pipeline."""
        assert trading_system._data_pipeline is None

        # Access property
        pipeline = trading_system.data_pipeline

        assert pipeline is not None
        assert trading_system._data_pipeline is not None

    def test_lazy_loading_risk_manager(self, trading_system):
        """Test lazy loading of risk manager."""
        assert trading_system._risk_manager is None

        # Access property
        rm = trading_system.risk_manager

        assert rm is not None
        assert trading_system._risk_manager is not None


# ============================================================================
# Data Loading Tests
# ============================================================================

class TestDataLoading:
    """Tests for data loading functionality."""

    def test_load_data_success(self, trading_system):
        """Test successful data loading."""
        with patch.object(trading_system, 'data_pipeline', MockDataPipeline()):
            market_data = trading_system.load_data(
                symbol='EURUSD',
                timeframe='1h',
                n_bars=1000
            )

            assert market_data is not None
            assert len(market_data.close) == 1000

    def test_load_data_connects_pipeline(self, trading_system):
        """Test that load_data connects the pipeline."""
        mock_pipeline = MockDataPipeline()

        with patch.object(trading_system, 'data_pipeline', mock_pipeline):
            trading_system.load_data('EURUSD', '1h', 1000)

            assert mock_pipeline.connected

    def test_load_data_returns_none_on_failure(self, trading_system):
        """Test that load_data returns None on failure."""
        mock_pipeline = Mock()
        mock_pipeline.connect = Mock()
        mock_pipeline.fetch_historical_data = Mock(return_value=None)

        with patch.object(trading_system, 'data_pipeline', mock_pipeline):
            result = trading_system.load_data('INVALID', '1h', 1000)

            assert result is None


# ============================================================================
# Model Initialization Tests
# ============================================================================

class TestModelInitialization:
    """Tests for model initialization."""

    def test_initialize_models(self, trading_system):
        """Test model initialization."""
        with patch('main.TransformerPredictor', MockPredictor):
            with patch('main.PPOAgent', MockAgent):
                trading_system.initialize_models(input_dim=55, state_dim=100)

                assert trading_system._predictor is not None
                assert trading_system._agent is not None

    def test_initialize_models_uses_config(self, trading_system):
        """Test that model initialization uses config values."""
        with patch('main.TransformerPredictor') as mock_pred_cls:
            with patch('main.PPOAgent') as mock_agent_cls:
                mock_pred_cls.return_value = MockPredictor()
                mock_agent_cls.return_value = MockAgent()

                trading_system.initialize_models(input_dim=55, state_dim=100)

                # Verify predictor was called with config values
                mock_pred_cls.assert_called_once()
                call_kwargs = mock_pred_cls.call_args
                assert call_kwargs[1]['input_dim'] == 55


# ============================================================================
# Environment Creation Tests
# ============================================================================

class TestEnvironmentCreation:
    """Tests for environment creation."""

    def test_create_environment(self, trading_system):
        """Test trading environment creation."""
        market_data = MockMarketData(n_bars=500)

        env = trading_system.create_environment(market_data)

        assert env is not None
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')

    def test_create_environment_correct_shape(self, trading_system):
        """Test that environment has correct observation shape."""
        market_data = MockMarketData(n_bars=500)

        env = trading_system.create_environment(market_data)

        # Should have observation space
        assert env.observation_space.shape[0] > 0


# ============================================================================
# Save/Load Model Tests
# ============================================================================

class TestSaveLoadModels:
    """Tests for model save/load functionality."""

    def test_save_models_creates_files(self, trading_system, temp_dir):
        """Test that save_models creates necessary files."""
        # Initialize mock models
        trading_system._predictor = MockPredictor()
        trading_system._agent = MockAgent()

        save_dir = os.path.join(temp_dir, 'test_models')
        trading_system.save_models(save_dir)

        assert os.path.exists(os.path.join(save_dir, 'predictor.pt'))
        assert os.path.exists(os.path.join(save_dir, 'agent.pt'))
        assert os.path.exists(os.path.join(save_dir, 'model_metadata.json'))
        assert os.path.exists(os.path.join(save_dir, 'config.json'))

    def test_save_models_metadata_content(self, trading_system, temp_dir):
        """Test that save_models creates correct metadata."""
        trading_system._predictor = MockPredictor(input_dim=55)
        trading_system._agent = MockAgent(state_dim=100, action_dim=4)

        save_dir = os.path.join(temp_dir, 'test_models')
        trading_system.save_models(save_dir)

        with open(os.path.join(save_dir, 'model_metadata.json')) as f:
            metadata = json.load(f)

        assert metadata['predictor']['input_dim'] == 55
        assert metadata['predictor']['exists'] is True
        assert metadata['agent']['state_dim'] == 100
        assert metadata['agent']['action_dim'] == 4
        assert metadata['agent']['exists'] is True

    def test_load_models_restores_state(self, trading_system, temp_dir):
        """Test that load_models restores models correctly."""
        # First save models
        trading_system._predictor = MockPredictor(input_dim=55)
        trading_system._agent = MockAgent(state_dim=100, action_dim=4)

        save_dir = os.path.join(temp_dir, 'test_models')
        trading_system.save_models(save_dir)

        # Clear models
        trading_system._predictor = None
        trading_system._agent = None

        # Load with mocked classes
        with patch('main.TransformerPredictor', MockPredictor):
            with patch('main.PPOAgent', MockAgent):
                trading_system.load_models(save_dir)

        assert trading_system._predictor is not None
        assert trading_system._agent is not None

    def test_load_models_missing_metadata(self, trading_system, temp_dir):
        """Test load_models handles missing metadata gracefully."""
        save_dir = os.path.join(temp_dir, 'empty_models')
        os.makedirs(save_dir, exist_ok=True)

        # Should not raise
        trading_system.load_models(save_dir)

        # Models should remain None
        assert trading_system._predictor is None
        assert trading_system._agent is None

    def test_save_predictor_only(self, trading_system, temp_dir):
        """Test _save_predictor_only saves predictor and preserves agent metadata."""
        # First save agent to create existing metadata
        trading_system._agent = MockAgent(state_dim=100, action_dim=4)
        save_dir = os.path.join(temp_dir, 'partial_save')
        trading_system._save_agent_only(save_dir)

        # Now save predictor only
        trading_system._predictor = MockPredictor(input_dim=55)
        trading_system._save_predictor_only(save_dir)

        # Verify both are in metadata
        with open(os.path.join(save_dir, 'model_metadata.json')) as f:
            metadata = json.load(f)

        assert metadata['predictor']['exists'] is True
        assert metadata['predictor']['input_dim'] == 55
        # Agent metadata should be preserved
        assert metadata['agent']['exists'] is True
        assert metadata['agent']['state_dim'] == 100

    def test_save_agent_only(self, trading_system, temp_dir):
        """Test _save_agent_only saves agent and preserves predictor metadata."""
        # First save predictor to create existing metadata
        trading_system._predictor = MockPredictor(input_dim=55)
        save_dir = os.path.join(temp_dir, 'partial_save')
        trading_system._save_predictor_only(save_dir)

        # Now save agent only
        trading_system._agent = MockAgent(state_dim=100, action_dim=4)
        trading_system._save_agent_only(save_dir)

        # Verify both are in metadata
        with open(os.path.join(save_dir, 'model_metadata.json')) as f:
            metadata = json.load(f)

        assert metadata['agent']['exists'] is True
        assert metadata['agent']['state_dim'] == 100
        # Predictor metadata should be preserved
        assert metadata['predictor']['exists'] is True
        assert metadata['predictor']['input_dim'] == 55

    def test_save_predictor_only_creates_files(self, trading_system, temp_dir):
        """Test _save_predictor_only creates only predictor file."""
        trading_system._predictor = MockPredictor(input_dim=55)
        save_dir = os.path.join(temp_dir, 'predictor_only')

        trading_system._save_predictor_only(save_dir)

        assert os.path.exists(os.path.join(save_dir, 'predictor.pt'))
        assert os.path.exists(os.path.join(save_dir, 'model_metadata.json'))
        assert os.path.exists(os.path.join(save_dir, 'config.json'))
        # Agent file should not exist
        assert not os.path.exists(os.path.join(save_dir, 'agent.pt'))

    def test_save_agent_only_creates_files(self, trading_system, temp_dir):
        """Test _save_agent_only creates only agent file."""
        trading_system._agent = MockAgent(state_dim=100, action_dim=4)
        save_dir = os.path.join(temp_dir, 'agent_only')

        trading_system._save_agent_only(save_dir)

        assert os.path.exists(os.path.join(save_dir, 'agent.pt'))
        assert os.path.exists(os.path.join(save_dir, 'model_metadata.json'))
        assert os.path.exists(os.path.join(save_dir, 'config.json'))
        # Predictor file should not exist
        assert not os.path.exists(os.path.join(save_dir, 'predictor.pt'))


# ============================================================================
# Training Tests
# ============================================================================

class TestTraining:
    """Tests for training functionality."""

    def test_prepare_training_data(self, trading_system):
        """Test training data preparation."""
        with patch.object(trading_system, 'data_pipeline', MockDataPipeline()):
            market_data = MockMarketData(n_bars=1000)

            splits, input_dim = trading_system.prepare_training_data(market_data)

            assert 'train' in splits
            assert 'val' in splits
            assert 'test' in splits
            assert input_dim > 0

    def test_train_integration(self, trading_system):
        """Test training integration (mocked)."""
        market_data = MockMarketData(n_bars=500)

        with patch.object(trading_system, 'data_pipeline', MockDataPipeline()):
            with patch('main.TransformerPredictor', MockPredictor):
                with patch('main.PPOAgent', MockAgent):
                    with patch('main.ModelTrainer') as mock_trainer_cls:
                        mock_trainer = Mock()
                        mock_trainer.train_predictor.return_value = {
                            'train_losses': [0.5],
                            'val_losses': [0.6]
                        }
                        mock_trainer.train_agent.return_value = {
                            'episode_rewards': [100]
                        }
                        mock_trainer.save_all = Mock()
                        mock_trainer_cls.return_value = mock_trainer

                        results = trading_system.train(
                            market_data=market_data,
                            predictor_epochs=1,
                            agent_timesteps=100
                        )

                        assert 'predictor' in results
                        assert 'agent' in results

    def test_train_predictor_only(self, trading_system):
        """Test training only the predictor model."""
        market_data = MockMarketData(n_bars=500)

        # Set up the internal _data_pipeline directly
        trading_system._data_pipeline = MockDataPipeline()

        with patch('main.TransformerPredictor', MockPredictor):
            with patch('main.PPOAgent', MockAgent):
                with patch('main.ModelTrainer') as mock_trainer_cls:
                    mock_trainer = Mock()
                    mock_trainer.train_predictor.return_value = {
                        'train_losses': [0.5],
                        'val_losses': [0.6]
                    }
                    mock_trainer_cls.return_value = mock_trainer

                    results = trading_system.train_predictor_only(
                        market_data=market_data,
                        predictor_epochs=1
                    )

                    # Verify only predictor training was called
                    mock_trainer.train_predictor.assert_called_once()
                    mock_trainer.train_agent.assert_not_called()

                    assert results['predictor'] is not None
                    assert results['agent'] is None

    def test_train_agent_only(self, trading_system):
        """Test training only the PPO agent model."""
        market_data = MockMarketData(n_bars=500)

        # Set up the internal _data_pipeline directly
        trading_system._data_pipeline = MockDataPipeline()

        with patch('main.TransformerPredictor', MockPredictor):
            with patch('main.PPOAgent', MockAgent):
                with patch('main.ModelTrainer') as mock_trainer_cls:
                    mock_trainer = Mock()
                    mock_trainer.train_agent.return_value = {
                        'episode_rewards': [100]
                    }
                    mock_trainer_cls.return_value = mock_trainer

                    results = trading_system.train_agent_only(
                        market_data=market_data,
                        agent_timesteps=100
                    )

                    # Verify only agent training was called
                    mock_trainer.train_agent.assert_called_once()
                    mock_trainer.train_predictor.assert_not_called()

                    assert results['predictor'] is None
                    assert results['agent'] is not None


# ============================================================================
# Backtest Tests
# ============================================================================

class TestBacktest:
    """Tests for backtesting functionality."""

    def test_backtest_basic(self, trading_system):
        """Test basic backtesting."""
        market_data = MockMarketData(n_bars=500)

        with patch('main.Backtester') as mock_backtester_cls:
            mock_backtester = Mock()
            mock_result = Mock()
            mock_result.total_return = 0.05
            mock_result.sharpe_ratio = 1.2
            mock_result.max_drawdown = 0.08
            mock_result.total_trades = 10
            mock_result.win_rate = 0.6
            mock_result.trades = []
            mock_backtester.run.return_value = mock_result
            mock_backtester_cls.return_value = mock_backtester

            with patch('main.PerformanceAnalyzer') as mock_analyzer_cls:
                mock_analyzer = Mock()
                mock_analyzer.analyze.return_value = {
                    'total_return': 0.05,
                    'sharpe_ratio': 1.2
                }
                mock_analyzer.generate_report.return_value = "Test report"
                mock_analyzer_cls.return_value = mock_analyzer

                result, analysis = trading_system.backtest(market_data)

                assert result is not None
                assert analysis is not None


# ============================================================================
# CLI Argument Parsing Tests
# ============================================================================

class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_train_command_args(self):
        """Test train command argument parsing."""
        test_args = ['train', '--symbol', 'GBPUSD', '--epochs', '50']

        with patch.object(sys, 'argv', ['main.py'] + test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument('command', choices=['train', 'backtest', 'evaluate', 'walkforward', 'autotrade'])
            parser.add_argument('--symbol', '-s', default='EURUSD')
            parser.add_argument('--epochs', '-e', type=int, default=100)

            args = parser.parse_args(test_args)

            assert args.command == 'train'
            assert args.symbol == 'GBPUSD'
            assert args.epochs == 50

    def test_train_model_type_args(self):
        """Test train command with --model-type argument."""
        # Test transformer only
        test_args = ['train', '--model-type', 'transformer', '--epochs', '50']
        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=['train', 'backtest', 'evaluate', 'walkforward', 'autotrade'])
        parser.add_argument('--model-type', choices=['transformer', 'ppo', 'both'], default='both')
        parser.add_argument('--epochs', '-e', type=int, default=100)

        args = parser.parse_args(test_args)
        assert args.model_type == 'transformer'

        # Test ppo only
        test_args = ['train', '--model-type', 'ppo', '--timesteps', '10000']
        parser.add_argument('--timesteps', type=int, default=100000)
        args = parser.parse_args(test_args)
        assert args.model_type == 'ppo'

        # Test both (default)
        test_args = ['train']
        args = parser.parse_args(test_args)
        assert args.model_type == 'both'

    def test_backtest_command_args(self):
        """Test backtest command argument parsing."""
        test_args = ['backtest', '--symbol', 'EURUSD', '--bars', '10000', '--realistic']

        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=['train', 'backtest', 'evaluate', 'walkforward', 'autotrade'])
        parser.add_argument('--symbol', '-s', default='EURUSD')
        parser.add_argument('--bars', '-b', type=int, default=50000)
        parser.add_argument('--realistic', action='store_true')

        args = parser.parse_args(test_args)

        assert args.command == 'backtest'
        assert args.bars == 10000
        assert args.realistic is True

    def test_autotrade_command_args(self):
        """Test autotrade command argument parsing."""
        test_args = ['autotrade', '--paper']

        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=['train', 'backtest', 'evaluate', 'walkforward', 'autotrade'])
        parser.add_argument('--paper', action='store_true')

        args = parser.parse_args(test_args)

        assert args.command == 'autotrade'
        assert args.paper is True

    def test_mlflow_args(self):
        """Test MLflow argument parsing."""
        test_args = ['train', '--no-mlflow', '--mlflow-experiment', 'test_exp']

        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=['train', 'backtest', 'evaluate', 'walkforward', 'autotrade'])
        parser.add_argument('--no-mlflow', action='store_true')
        parser.add_argument('--mlflow-experiment', default=None)

        args = parser.parse_args(test_args)

        assert args.no_mlflow is True
        assert args.mlflow_experiment == 'test_exp'


# ============================================================================
# Logging Initialization Tests
# ============================================================================

class TestLoggingInitialization:
    """Tests for logging initialization."""

    def test_initialize_logging_default(self, mock_config):
        """Test logging initialization with defaults."""
        with patch('main.setup_logging') as mock_setup:
            initialize_logging(mock_config)

            mock_setup.assert_called_once()

    def test_initialize_logging_with_override(self, mock_config):
        """Test logging initialization with CLI override."""
        with patch('main.setup_logging') as mock_setup:
            initialize_logging(
                mock_config,
                log_level_override='DEBUG'
            )

            mock_setup.assert_called_once()
            call_kwargs = mock_setup.call_args[1]
            assert call_kwargs['level'] == 'DEBUG'

    def test_initialize_logging_with_file_override(self, mock_config, temp_dir):
        """Test logging initialization with file override."""
        log_file = os.path.join(temp_dir, 'test.log')

        with patch('main.setup_logging') as mock_setup:
            initialize_logging(
                mock_config,
                log_file_override=log_file
            )

            mock_setup.assert_called_once()
            call_kwargs = mock_setup.call_args[1]
            assert call_kwargs['log_file'] == log_file


# ============================================================================
# Main Function Tests
# ============================================================================

class TestMainFunction:
    """Tests for main() function."""

    def test_main_train_command(self, temp_dir):
        """Test main() with train command."""
        test_args = [
            'train',
            '--symbol', 'EURUSD',
            '--epochs', '1',
            '--timesteps', '100',
            '--bars', '500',
            '--model-dir', os.path.join(temp_dir, 'models'),
            '--no-mlflow'
        ]

        with patch.object(sys, 'argv', ['main.py'] + test_args):
            with patch('main.LeapTradingSystem') as mock_system_cls:
                mock_system = Mock()
                mock_system.load_data.return_value = MockMarketData(500)
                mock_system.train.return_value = {
                    'predictor': {'train_losses': [0.5]},
                    'agent': {'episode_rewards': [100]}
                }
                mock_system.save_models = Mock()
                mock_system_cls.return_value = mock_system

                with patch('cli.parser.get_config') as mock_get_config:
                    config = get_config()
                    config.base_dir = temp_dir
                    config.mlflow.enabled = False
                    mock_get_config.return_value = config

                    with patch('main.initialize_logging'):
                        # Main should complete without error
                        try:
                            main()
                        except SystemExit as e:
                            # May exit with 0 on success
                            assert e.code in [0, None]

    def test_main_backtest_command(self, temp_dir):
        """Test main() with backtest command."""
        test_args = [
            'backtest',
            '--symbol', 'EURUSD',
            '--bars', '500',
            '--model-dir', os.path.join(temp_dir, 'models'),
            '--no-mlflow'
        ]

        with patch.object(sys, 'argv', ['main.py'] + test_args):
            with patch('main.LeapTradingSystem') as mock_system_cls:
                mock_system = Mock()
                mock_system.load_data.return_value = MockMarketData(500)
                mock_result = Mock()
                mock_result.trades = []
                mock_system.backtest.return_value = (mock_result, {
                    'total_return': 0.05,
                    'sharpe_ratio': 1.2
                })
                mock_system.load_models = Mock()
                mock_system.mlflow_tracker = None
                mock_system_cls.return_value = mock_system

                with patch('cli.parser.get_config') as mock_get_config:
                    config = get_config()
                    config.base_dir = temp_dir
                    config.mlflow.enabled = False
                    mock_get_config.return_value = config

                    with patch('main.initialize_logging'):
                        try:
                            main()
                        except SystemExit as e:
                            assert e.code in [0, None]


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_load_data_failure_exits(self, trading_system):
        """Test that data loading failure is handled."""
        with patch.object(trading_system, 'data_pipeline') as mock_pipeline:
            mock_pipeline.connect = Mock()
            mock_pipeline.fetch_historical_data.return_value = None

            result = trading_system.load_data('INVALID', '1h', 1000)

            assert result is None

    def test_save_models_without_predictor(self, trading_system, temp_dir):
        """Test save_models with only agent."""
        trading_system._predictor = None
        trading_system._agent = MockAgent()

        save_dir = os.path.join(temp_dir, 'partial_models')
        trading_system.save_models(save_dir)

        with open(os.path.join(save_dir, 'model_metadata.json')) as f:
            metadata = json.load(f)

        assert 'predictor' not in metadata or not metadata.get('predictor', {}).get('exists', False)
        assert metadata['agent']['exists'] is True

    def test_save_models_without_agent(self, trading_system, temp_dir):
        """Test save_models with only predictor."""
        trading_system._predictor = MockPredictor()
        trading_system._agent = None

        save_dir = os.path.join(temp_dir, 'partial_models')
        trading_system.save_models(save_dir)

        with open(os.path.join(save_dir, 'model_metadata.json')) as f:
            metadata = json.load(f)

        assert metadata['predictor']['exists'] is True
        assert 'agent' not in metadata or not metadata.get('agent', {}).get('exists', False)


# ============================================================================
# Modular Config Loader Tests
# ============================================================================

class TestModularConfigLoaders:
    """Tests for modular config loading functions."""

    def test_load_training_config(self, temp_dir):
        """Test loading standalone training config."""
        config_data = {
            "device": "cpu",
            "seed": 123,
            "transformer": {
                "d_model": 64,
                "n_heads": 4,
                "learning_rate": 1e-5,
                "epochs": 50
            },
            "ppo": {
                "learning_rate": 1e-4,
                "gamma": 0.95,
                "total_timesteps": 500000
            }
        }
        config_path = os.path.join(temp_dir, 'training.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        transformer_cfg, ppo_cfg, device, seed = load_training_config(config_path)

        assert device == "cpu"
        assert seed == 123
        assert transformer_cfg.d_model == 64
        assert transformer_cfg.n_heads == 4
        assert transformer_cfg.learning_rate == 1e-5
        assert transformer_cfg.epochs == 50
        assert ppo_cfg.learning_rate == 1e-4
        assert ppo_cfg.gamma == 0.95
        assert ppo_cfg.total_timesteps == 500000

    def test_load_training_config_defaults(self, temp_dir):
        """Test training config uses defaults for missing fields."""
        config_data = {
            "transformer": {
                "d_model": 256
            },
            "ppo": {
                "gamma": 0.98
            }
        }
        config_path = os.path.join(temp_dir, 'training.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        transformer_cfg, ppo_cfg, device, seed = load_training_config(config_path)

        # Check that defaults are used for missing fields
        assert device == "auto"  # default
        assert seed == 42  # default
        assert transformer_cfg.d_model == 256  # from config
        assert transformer_cfg.n_heads == 8  # default
        assert ppo_cfg.gamma == 0.98  # from config
        assert ppo_cfg.learning_rate == 3e-4  # default

    def test_load_data_config(self, temp_dir):
        """Test loading standalone data config."""
        config_data = {
            "symbols": ["GBPUSD", "USDJPY"],
            "primary_timeframe": "4h",
            "lookback_window": 100,
            "use_technical_indicators": False
        }
        config_path = os.path.join(temp_dir, 'data.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        data_cfg = load_data_config(config_path)

        assert data_cfg.symbols == ["GBPUSD", "USDJPY"]
        assert data_cfg.primary_timeframe == "4h"
        assert data_cfg.lookback_window == 100
        assert data_cfg.use_technical_indicators is False

    def test_load_backtest_config(self, temp_dir):
        """Test loading standalone backtest config."""
        config_data = {
            "initial_balance": 50000.0,
            "leverage": 50,
            "commission_per_lot": 5.0
        }
        config_path = os.path.join(temp_dir, 'backtest.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        backtest_cfg = load_backtest_config(config_path)

        assert backtest_cfg.initial_balance == 50000.0
        assert backtest_cfg.leverage == 50
        assert backtest_cfg.commission_per_lot == 5.0

    def test_load_risk_config(self, temp_dir):
        """Test loading standalone risk config."""
        config_data = {
            "max_position_size": 0.05,
            "max_daily_loss": 0.10,
            "max_drawdown": 0.25
        }
        config_path = os.path.join(temp_dir, 'risk.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        risk_cfg = load_risk_config(config_path)

        assert risk_cfg.max_position_size == 0.05
        assert risk_cfg.max_daily_loss == 0.10
        assert risk_cfg.max_drawdown == 0.25

    def test_load_auto_trader_config(self, temp_dir):
        """Test loading standalone auto-trader config."""
        config_data = {
            "symbols": ["EURUSD", "GBPUSD"],
            "timeframe": "4h",
            "risk_per_trade": 0.02,
            "max_positions": 5
        }
        config_path = os.path.join(temp_dir, 'auto_trader.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        auto_trader_cfg = load_auto_trader_config(config_path)

        assert auto_trader_cfg.symbols == ["EURUSD", "GBPUSD"]
        assert auto_trader_cfg.timeframe == "4h"
        assert auto_trader_cfg.risk_per_trade == 0.02
        assert auto_trader_cfg.max_positions == 5

    def test_load_logging_config(self, temp_dir):
        """Test loading standalone logging config."""
        config_data = {
            "level": "DEBUG",
            "log_to_file": False,
            "max_file_size_mb": 20
        }
        config_path = os.path.join(temp_dir, 'logging.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        logging_cfg = load_logging_config(config_path)

        assert logging_cfg.level == "DEBUG"
        assert logging_cfg.log_to_file is False
        assert logging_cfg.max_file_size_mb == 20

    def test_load_config_file_not_found(self, temp_dir):
        """Test that loading non-existent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_training_config(os.path.join(temp_dir, 'nonexistent.json'))

    def test_load_config_invalid_json(self, temp_dir):
        """Test that loading invalid JSON raises JSONDecodeError."""
        config_path = os.path.join(temp_dir, 'invalid.json')
        with open(config_path, 'w') as f:
            f.write("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            load_training_config(config_path)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
