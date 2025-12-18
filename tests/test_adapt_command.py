"""
Tests for adapt CLI command.

Tests the execute_adapt command handler and related functionality
for online model adaptation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from argparse import Namespace
from dataclasses import dataclass

from training.online_learning import AdaptationConfig


class TestAdaptationConfig:
    """Tests for AdaptationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AdaptationConfig()

        assert config.error_threshold == 0.05
        assert config.drawdown_threshold == 0.1
        assert config.adaptation_frequency == 100
        assert config.max_adaptations_per_day == 10
        assert config.min_samples_for_adaptation == 50

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AdaptationConfig(
            error_threshold=0.03,
            drawdown_threshold=0.05,
            adaptation_frequency=50,
            max_adaptations_per_day=5,
            min_samples_for_adaptation=100
        )

        assert config.error_threshold == 0.03
        assert config.drawdown_threshold == 0.05
        assert config.adaptation_frequency == 50
        assert config.max_adaptations_per_day == 5
        assert config.min_samples_for_adaptation == 100


class TestExecuteAdaptCommand:
    """Tests for execute_adapt command."""

    @pytest.fixture
    def mock_system(self):
        """Create mock LeapTradingSystem."""
        system = Mock()
        system._predictor = Mock()
        system._agent = Mock()
        system.load_models = Mock()
        system.load_data = Mock()
        return system

    @pytest.fixture
    def mock_config(self):
        """Create mock SystemConfig."""
        config = Mock()
        config.data = Mock()
        config.data.lookback_window = 60
        config.backtest = Mock()
        config.backtest.initial_balance = 10000.0
        return config

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        n_bars = 500
        np.random.seed(42)

        close = np.cumsum(np.random.randn(n_bars) * 0.001) + 1.10
        high = close + np.random.uniform(0.0001, 0.001, n_bars)
        low = close - np.random.uniform(0.0001, 0.001, n_bars)
        open_ = low + (high - low) * np.random.uniform(0.3, 0.7, n_bars)

        market_data = Mock()
        market_data.open = open_
        market_data.high = high
        market_data.low = low
        market_data.close = close
        market_data.volume = np.random.randint(1000, 5000, n_bars).astype(float)
        market_data.features = np.random.randn(n_bars, 20)
        market_data.feature_names = [f'feature_{i}' for i in range(20)]
        return market_data

    def test_adapt_args_parsing(self):
        """Test adapt command argument parsing."""
        # Create args as would be parsed from CLI
        args = Namespace(
            mode='offline',
            adapt_bars=10000,
            adapt_epochs=10,
            adapt_timesteps=10000,
            error_threshold=0.05,
            drawdown_threshold=0.1,
            adapt_frequency=100,
            max_adaptations=10,
            min_samples=50,
            model_dir='./saved_models',
            save=False
        )

        # Verify args exist
        assert args.mode == 'offline'
        assert args.adapt_bars == 10000
        assert args.adapt_epochs == 10
        assert args.adapt_timesteps == 10000
        assert args.error_threshold == 0.05
        assert args.save is False

    def test_adapt_config_from_args(self):
        """Test creating AdaptationConfig from CLI args."""
        args = Namespace(
            error_threshold=0.03,
            drawdown_threshold=0.08,
            adapt_frequency=75,
            max_adaptations=5,
            min_samples=100
        )

        config = AdaptationConfig(
            error_threshold=getattr(args, 'error_threshold', 0.05),
            drawdown_threshold=getattr(args, 'drawdown_threshold', 0.1),
            adaptation_frequency=getattr(args, 'adapt_frequency', 100),
            max_adaptations_per_day=getattr(args, 'max_adaptations', 10),
            min_samples_for_adaptation=getattr(args, 'min_samples', 50)
        )

        assert config.error_threshold == 0.03
        assert config.drawdown_threshold == 0.08
        assert config.adaptation_frequency == 75
        assert config.max_adaptations_per_day == 5
        assert config.min_samples_for_adaptation == 100

    @patch('cli.commands.adapt.AdaptiveTrainer')
    def test_execute_adapt_offline_mode(
        self, mock_trainer_class, mock_system, mock_config, sample_market_data
    ):
        """Test execute_adapt in offline mode."""
        from cli.commands.adapt import execute_adapt

        mock_system.load_data.return_value = sample_market_data
        mock_trainer = Mock()
        mock_trainer.train_offline.return_value = {
            'predictor': {'loss': 0.001},
            'agent': {'reward': 100}
        }
        mock_trainer_class.return_value = mock_trainer

        args = Namespace(
            mode='offline',
            adapt_bars=500,
            adapt_epochs=5,
            adapt_timesteps=5000,
            error_threshold=0.05,
            drawdown_threshold=0.1,
            adapt_frequency=100,
            max_adaptations=10,
            min_samples=50,
            model_dir='./saved_models',
            save=False
        )

        resolved = {
            'symbols': ['EURUSD'],
            'timeframe': '1h'
        }

        # Should not raise
        execute_adapt(mock_system, args, mock_config, resolved)

        # Verify models were loaded
        mock_system.load_models.assert_called_once()

        # Verify data was loaded
        mock_system.load_data.assert_called()

    def test_execute_adapt_no_models_loaded(self, mock_system, mock_config):
        """Test execute_adapt when no models are loaded."""
        from cli.commands.adapt import execute_adapt

        mock_system._predictor = None
        mock_system._agent = None

        args = Namespace(
            mode='offline',
            adapt_bars=500,
            model_dir='./saved_models',
            error_threshold=0.05,
            drawdown_threshold=0.1,
            adapt_frequency=100,
            max_adaptations=10,
            min_samples=50
        )

        resolved = {
            'symbols': ['EURUSD'],
            'timeframe': '1h'
        }

        # Should handle gracefully (log error and return)
        execute_adapt(mock_system, args, mock_config, resolved)

    def test_execute_adapt_load_models_fails(self, mock_system, mock_config):
        """Test execute_adapt when model loading fails."""
        from cli.commands.adapt import execute_adapt

        mock_system.load_models.side_effect = Exception("Model not found")
        mock_system._predictor = None
        mock_system._agent = None

        args = Namespace(
            mode='offline',
            adapt_bars=500,
            model_dir='./nonexistent',
            error_threshold=0.05,
            drawdown_threshold=0.1,
            adapt_frequency=100,
            max_adaptations=10,
            min_samples=50
        )

        resolved = {
            'symbols': ['EURUSD'],
            'timeframe': '1h'
        }

        # Should handle gracefully
        execute_adapt(mock_system, args, mock_config, resolved)

    @patch('cli.commands.adapt.AdaptiveTrainer')
    def test_execute_adapt_multiple_symbols(
        self, mock_trainer_class, mock_system, mock_config, sample_market_data
    ):
        """Test execute_adapt with multiple symbols."""
        from cli.commands.adapt import execute_adapt

        mock_system.load_data.return_value = sample_market_data
        mock_trainer = Mock()
        mock_trainer.train_offline.return_value = {
            'predictor': {'loss': 0.001},
            'agent': {'reward': 100}
        }
        mock_trainer_class.return_value = mock_trainer

        args = Namespace(
            mode='offline',
            adapt_bars=500,
            adapt_epochs=5,
            adapt_timesteps=5000,
            error_threshold=0.05,
            drawdown_threshold=0.1,
            adapt_frequency=100,
            max_adaptations=10,
            min_samples=50,
            model_dir='./saved_models',
            save=False
        )

        resolved = {
            'symbols': ['EURUSD', 'GBPUSD'],
            'timeframe': '1h'
        }

        execute_adapt(mock_system, args, mock_config, resolved)

        # Should load data for each symbol
        assert mock_system.load_data.call_count == 2


class TestPrepareSequences:
    """Tests for _prepare_sequences helper function."""

    def test_prepare_sequences(self):
        """Test sequence preparation for predictor training."""
        from cli.commands.adapt import _prepare_sequences

        n_samples = 100
        n_features = 10
        lookback = 20

        features = np.random.randn(n_samples, n_features)
        close = np.cumsum(np.random.randn(n_samples) * 0.001) + 1.10

        X, y = _prepare_sequences(features, lookback, close)

        # Expected number of samples: n_samples - lookback - 1
        expected_samples = n_samples - lookback - 1

        assert X.shape == (expected_samples, lookback, n_features)
        assert y.shape == (expected_samples,)

    def test_prepare_sequences_insufficient_data(self):
        """Test sequence preparation with insufficient data."""
        from cli.commands.adapt import _prepare_sequences

        features = np.random.randn(10, 5)  # Only 10 samples
        close = np.random.randn(10)
        lookback = 20  # Lookback > samples

        X, y = _prepare_sequences(features, lookback, close)

        # Should return empty arrays
        assert len(X) == 0
        assert len(y) == 0


class TestEvaluateAdaptation:
    """Tests for _evaluate_adaptation helper function."""

    @pytest.fixture
    def mock_system(self):
        """Create mock system with predictor and agent."""
        system = Mock()

        # Mock predictor
        system._predictor = Mock()
        system._predictor.predict.return_value = {
            'prediction': np.array([[0.001]])
        }

        # Mock agent
        system._agent = Mock()
        system._agent.select_action.return_value = (1, 0.8, None)

        return system

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        n_bars = 100
        np.random.seed(42)

        market_data = Mock()
        market_data.close = np.cumsum(np.random.randn(n_bars) * 0.001) + 1.10
        market_data.features = np.random.randn(n_bars, 20)
        market_data.feature_names = [f'feature_{i}' for i in range(20)]
        return market_data

    @pytest.fixture
    def mock_config(self):
        """Create mock config for evaluation tests."""
        config = Mock()
        config.data = Mock()
        config.data.lookback_window = 60
        return config

    @patch('cli.commands.adapt.logger')
    def test_evaluate_adaptation(
        self, mock_logger, mock_system, sample_market_data, mock_config
    ):
        """Test adaptation evaluation."""
        from cli.commands.adapt import _evaluate_adaptation

        # Should not raise
        _evaluate_adaptation(mock_system, sample_market_data, 'EURUSD', mock_config)

        # Should log evaluation info
        assert mock_logger.info.called

    @patch('cli.commands.adapt.logger')
    def test_evaluate_adaptation_insufficient_data(self, mock_logger, mock_system, mock_config):
        """Test evaluation with insufficient data."""
        from cli.commands.adapt import _evaluate_adaptation

        market_data = Mock()
        market_data.features = np.random.randn(10, 5)  # Too few samples
        market_data.close = np.random.randn(10)

        _evaluate_adaptation(mock_system, market_data, 'EURUSD', mock_config)

        # Should log error about insufficient data
        assert mock_logger.error.called


class TestAdaptResultsFormatting:
    """Tests for result formatting functions."""

    @patch('cli.commands.adapt.logger')
    def test_print_adaptation_results(self, mock_logger):
        """Test adaptation results printing."""
        from cli.commands.adapt import _print_adaptation_results

        results = {
            'predictor': {
                'final_loss': 0.001,
                'epochs_trained': 10
            },
            'agent': {
                'mean_reward': 100.5,
                'timesteps': 5000
            }
        }

        _print_adaptation_results(results)

        # Should log results
        assert mock_logger.info.called

    @patch('cli.commands.adapt.logger')
    def test_print_performance_report(self, mock_logger):
        """Test performance report printing."""
        from cli.commands.adapt import _print_performance_report

        report = {
            'total_steps': 1000,
            'total_adaptations': 5,
            'current_regime': 'trending',
            'avg_prediction_error': 0.003,
            'total_pnl': 500.0,
            'avg_sharpe': 1.5,
            'avg_win_rate': 0.55,
            'regime_distribution': {
                'trending': 0.6,
                'ranging': 0.4
            }
        }

        _print_performance_report(report)

        # Should log report
        assert mock_logger.info.called

    @patch('cli.commands.adapt.logger')
    def test_print_empty_report(self, mock_logger):
        """Test printing empty performance report."""
        from cli.commands.adapt import _print_performance_report

        _print_performance_report({})

        # Should handle empty report
        assert mock_logger.info.called


class TestAdaptCommandIntegration:
    """Integration tests for adapt command with CLI."""

    def test_adapt_in_command_registry(self):
        """Test adapt command is registered."""
        from cli.commands import COMMANDS

        assert 'adapt' in COMMANDS
        assert callable(COMMANDS['adapt'])

    def test_adapt_parser_has_mode_argument(self):
        """Test parser includes mode argument."""
        from cli.parser import create_parser

        parser = create_parser()
        args = parser.parse_args(['adapt', '--mode', 'offline'])

        assert args.command == 'adapt'
        assert args.mode == 'offline'

    def test_adapt_parser_has_all_arguments(self):
        """Test parser includes all adapt arguments."""
        from cli.parser import create_parser

        parser = create_parser()
        args = parser.parse_args([
            'adapt',
            '--mode', 'evaluate',
            '--adapt-bars', '5000',
            '--adapt-epochs', '15',
            '--adapt-timesteps', '20000',
            '--error-threshold', '0.03',
            '--drawdown-threshold', '0.08',
            '--adapt-frequency', '50',
            '--max-adaptations', '5',
            '--min-samples', '100',
            '--save'
        ])

        assert args.mode == 'evaluate'
        assert args.adapt_bars == 5000
        assert args.adapt_epochs == 15
        assert args.adapt_timesteps == 20000
        assert args.error_threshold == 0.03
        assert args.drawdown_threshold == 0.08
        assert args.adapt_frequency == 50
        assert args.max_adaptations == 5
        assert args.min_samples == 100
        assert args.save is True

    def test_adapt_mode_choices(self):
        """Test adapt mode has correct choices."""
        from cli.parser import create_parser

        parser = create_parser()

        # Valid modes should work
        for mode in ['offline', 'online', 'evaluate']:
            args = parser.parse_args(['adapt', '--mode', mode])
            assert args.mode == mode

        # Invalid mode should raise
        with pytest.raises(SystemExit):
            parser.parse_args(['adapt', '--mode', 'invalid'])
