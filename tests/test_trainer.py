"""
Leap Trading System - ModelTrainer Tests
Comprehensive tests for the training module.
"""

import pytest
import numpy as np
import torch
import os
import json
import tempfile
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime


# Import components to test
from training.trainer import ModelTrainer


# ============================================================================
# Mock Classes for Testing
# ============================================================================

class MockPredictor:
    """Mock TransformerPredictor for testing."""

    def __init__(self, input_dim=10):
        self.input_dim = input_dim
        self.model = Mock()
        self.trained = False
        self._saved_path = None
        self._loaded_path = None

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32,
              verbose=True, mlflow_callback=None):
        """Mock training that returns realistic results."""
        self.trained = True
        train_losses = [1.0 - i * 0.05 for i in range(epochs)]
        val_losses = [1.1 - i * 0.04 for i in range(epochs)]

        # Call the callback if provided
        if mlflow_callback:
            for epoch in range(epochs):
                mlflow_callback({
                    'train_loss': train_losses[epoch],
                    'val_loss': val_losses[epoch],
                    'learning_rate': 0.001
                }, epoch)

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': min(val_losses)
        }

    def predict(self, X, return_uncertainty=False):
        """Mock prediction."""
        batch_size = X.shape[0]
        return {
            'prediction': np.random.randn(batch_size, 1),
            'quantiles': np.random.randn(batch_size, 3),
            'uncertainty': np.abs(np.random.randn(batch_size))
        }

    def save(self, path):
        """Mock save."""
        self._saved_path = path
        # Create the file for existence checks
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('mock_model')

    def load(self, path):
        """Mock load."""
        self._loaded_path = path


class MockAgent:
    """Mock PPOAgent for testing."""

    def __init__(self, state_dim=20, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = Mock()
        self.trained = False
        self._saved_path = None
        self._loaded_path = None

    def train_on_env(self, env, total_timesteps=1000, eval_env=None,
                     eval_frequency=100, mlflow_callback=None):
        """Mock training on environment."""
        self.trained = True

        n_episodes = total_timesteps // 100
        episode_rewards = [np.random.uniform(-100, 200) for _ in range(n_episodes)]
        episode_lengths = [np.random.randint(50, 150) for _ in range(n_episodes)]

        # Call callback if provided
        if mlflow_callback:
            for step in range(0, total_timesteps, 100):
                mlflow_callback({
                    'policy_loss': np.random.uniform(0.1, 1.0),
                    'value_loss': np.random.uniform(0.1, 1.0),
                    'entropy': np.random.uniform(0.5, 1.5),
                    'episode_reward': np.random.uniform(-100, 200)
                }, step)

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }

    def select_action(self, state):
        """Mock action selection."""
        action = np.random.randint(0, self.action_dim)
        log_prob = np.random.uniform(-2, 0)
        value = np.random.uniform(-1, 1)
        return action, log_prob, value

    def evaluate(self, env, n_episodes=10):
        """Mock evaluation."""
        return np.random.uniform(0, 100)

    def save(self, path):
        """Mock save."""
        self._saved_path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('mock_agent')

    def load(self, path):
        """Mock load."""
        self._loaded_path = path


class MockDataPipeline:
    """Mock DataPipeline for testing."""

    def __init__(self):
        self.connected = False

    def connect(self):
        self.connected = True

    def prepare_sequences(self, data, sequence_length=60, prediction_horizon=12):
        """Return mock sequences."""
        n_samples = 100
        n_features = 10
        X = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
        y = np.random.randn(n_samples).astype(np.float32)
        timestamps = [datetime.now() for _ in range(n_samples)]
        return X, y, timestamps

    def create_train_val_test_split(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """Return mock splits."""
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return {
            'train': (X[:train_end], y[:train_end]),
            'val': (X[train_end:val_end], y[train_end:val_end]),
            'test': (X[val_end:], y[val_end:])
        }


class MockEnv:
    """Mock training environment."""

    def __init__(self, state_dim=20):
        self.state_dim = state_dim
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None):
        return np.random.randn(self.state_dim).astype(np.float32), {}

    def step(self, action):
        obs = np.random.randn(self.state_dim).astype(np.float32)
        reward = np.random.uniform(-1, 1)
        terminated = np.random.random() < 0.01
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class MockMarketData:
    """Mock market data for testing."""

    def __init__(self, n_bars=1000):
        self.close = np.random.uniform(1.0, 1.5, n_bars)
        self.open = self.close * (1 + np.random.uniform(-0.001, 0.001, n_bars))
        self.high = np.maximum(self.close, self.open) * (1 + np.random.uniform(0, 0.002, n_bars))
        self.low = np.minimum(self.close, self.open) * (1 - np.random.uniform(0, 0.002, n_bars))
        self.volume = np.random.uniform(1000, 10000, n_bars)
        self.features = np.random.randn(n_bars, 50).astype(np.float32)
        self.feature_names = [f'feature_{i}' for i in range(50)]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_predictor():
    """Create mock predictor."""
    return MockPredictor(input_dim=10)


@pytest.fixture
def mock_agent():
    """Create mock agent."""
    return MockAgent(state_dim=20, action_dim=4)


@pytest.fixture
def mock_data_pipeline():
    """Create mock data pipeline."""
    return MockDataPipeline()


@pytest.fixture
def mock_env():
    """Create mock environment."""
    return MockEnv(state_dim=20)


@pytest.fixture
def trainer(mock_predictor, mock_agent, mock_data_pipeline):
    """Create ModelTrainer with mocked components."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ModelTrainer(
            predictor=mock_predictor,
            agent=mock_agent,
            data_pipeline=mock_data_pipeline,
            config={
                'predictor_epochs': 5,
                'agent_timesteps': 500,
                'batch_size': 32,
                'checkpoint_dir': tmpdir
            },
            device='cpu'
        )
        yield trainer


@pytest.fixture
def training_data():
    """Create mock training data."""
    n_samples = 100
    seq_length = 60
    n_features = 10

    X_train = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)
    y_train = np.random.randn(n_samples).astype(np.float32)
    X_val = np.random.randn(20, seq_length, n_features).astype(np.float32)
    y_val = np.random.randn(20).astype(np.float32)

    return X_train, y_train, X_val, y_val


# ============================================================================
# ModelTrainer Initialization Tests
# ============================================================================

class TestModelTrainerInitialization:
    """Tests for ModelTrainer initialization."""

    def test_initialization_default_config(self, mock_predictor, mock_agent, mock_data_pipeline):
        """Test initialization with default configuration."""
        trainer = ModelTrainer(
            predictor=mock_predictor,
            agent=mock_agent,
            data_pipeline=mock_data_pipeline
        )

        assert trainer.predictor == mock_predictor
        assert trainer.agent == mock_agent
        assert trainer.data_pipeline == mock_data_pipeline
        assert trainer.predictor_epochs == 100  # Default
        assert trainer.agent_timesteps == 100000  # Default

    def test_initialization_custom_config(self, mock_predictor, mock_agent, mock_data_pipeline):
        """Test initialization with custom configuration."""
        config = {
            'predictor_epochs': 50,
            'agent_timesteps': 50000,
            'batch_size': 64
        }

        trainer = ModelTrainer(
            predictor=mock_predictor,
            agent=mock_agent,
            data_pipeline=mock_data_pipeline,
            config=config
        )

        assert trainer.predictor_epochs == 50
        assert trainer.agent_timesteps == 50000
        assert trainer.batch_size == 64

    def test_initialization_auto_device_cpu(self, mock_predictor, mock_agent, mock_data_pipeline):
        """Test auto device selection falls back to CPU."""
        trainer = ModelTrainer(
            predictor=mock_predictor,
            agent=mock_agent,
            data_pipeline=mock_data_pipeline,
            device='auto'
        )

        # Should be either cuda or cpu depending on availability
        assert trainer.device.type in ['cuda', 'cpu']

    def test_initialization_explicit_cpu(self, mock_predictor, mock_agent, mock_data_pipeline):
        """Test explicit CPU device selection."""
        trainer = ModelTrainer(
            predictor=mock_predictor,
            agent=mock_agent,
            data_pipeline=mock_data_pipeline,
            device='cpu'
        )

        assert trainer.device.type == 'cpu'

    def test_initialization_creates_checkpoint_dir(self, mock_predictor, mock_agent, mock_data_pipeline):
        """Test that checkpoint directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, 'checkpoints')

            trainer = ModelTrainer(
                predictor=mock_predictor,
                agent=mock_agent,
                data_pipeline=mock_data_pipeline,
                config={'checkpoint_dir': checkpoint_dir}
            )

            assert os.path.exists(checkpoint_dir)

    def test_initialization_empty_training_history(self, trainer):
        """Test that training history starts empty."""
        assert trainer.training_history['predictor'] == []
        assert trainer.training_history['agent'] == []
        assert trainer.training_history['combined'] == []


# ============================================================================
# Predictor Training Tests
# ============================================================================

class TestPredictorTraining:
    """Tests for predictor training."""

    def test_train_predictor_basic(self, trainer, training_data):
        """Test basic predictor training."""
        X_train, y_train, X_val, y_val = training_data

        results = trainer.train_predictor(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=3
        )

        assert 'train_losses' in results
        assert 'val_losses' in results
        assert len(results['train_losses']) == 3
        assert trainer.predictor.trained

    def test_train_predictor_uses_default_epochs(self, trainer, training_data):
        """Test that default epochs from config are used."""
        X_train, y_train, X_val, y_val = training_data

        results = trainer.train_predictor(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        # Should use config's predictor_epochs (5 from fixture)
        assert len(results['train_losses']) == 5

    def test_train_predictor_updates_history(self, trainer, training_data):
        """Test that training history is updated."""
        X_train, y_train, X_val, y_val = training_data

        trainer.train_predictor(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=3
        )

        assert len(trainer.training_history['predictor']) == 1
        history_entry = trainer.training_history['predictor'][0]
        assert 'timestamp' in history_entry
        assert 'epochs' in history_entry
        assert 'final_train_loss' in history_entry
        assert 'final_val_loss' in history_entry
        assert 'training_duration_seconds' in history_entry

    def test_train_predictor_saves_checkpoint(self, trainer, training_data):
        """Test that checkpoint is saved after training."""
        X_train, y_train, X_val, y_val = training_data

        trainer.train_predictor(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=2
        )

        # Check that predictor was saved
        assert trainer.predictor._saved_path is not None
        assert 'predictor_' in trainer.predictor._saved_path

    def test_train_predictor_with_callbacks(self, trainer, training_data):
        """Test training with callbacks."""
        X_train, y_train, X_val, y_val = training_data

        callback_called = []

        def test_callback(event, results):
            callback_called.append((event, results))

        trainer.train_predictor(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=2,
            callbacks=[test_callback]
        )

        assert len(callback_called) == 1
        assert callback_called[0][0] == 'predictor_trained'


# ============================================================================
# Agent Training Tests
# ============================================================================

class TestAgentTraining:
    """Tests for agent training."""

    def test_train_agent_basic(self, trainer, mock_env):
        """Test basic agent training."""
        results = trainer.train_agent(
            env=mock_env,
            total_timesteps=200
        )

        assert 'episode_rewards' in results
        assert 'episode_lengths' in results
        assert trainer.agent.trained

    def test_train_agent_uses_default_timesteps(self, trainer, mock_env):
        """Test that default timesteps from config are used."""
        results = trainer.train_agent(env=mock_env)

        # Should use config's agent_timesteps
        assert trainer.agent.trained

    def test_train_agent_updates_history(self, trainer, mock_env):
        """Test that training history is updated."""
        trainer.train_agent(env=mock_env, total_timesteps=200)

        assert len(trainer.training_history['agent']) == 1
        history_entry = trainer.training_history['agent'][0]
        assert 'timestamp' in history_entry
        assert 'timesteps' in history_entry
        assert 'final_reward' in history_entry
        assert 'total_episodes' in history_entry
        assert 'training_duration_seconds' in history_entry

    def test_train_agent_saves_checkpoint(self, trainer, mock_env):
        """Test that checkpoint is saved after training."""
        trainer.train_agent(env=mock_env, total_timesteps=200)

        assert trainer.agent._saved_path is not None
        assert 'agent_' in trainer.agent._saved_path

    def test_train_agent_with_callbacks(self, trainer, mock_env):
        """Test training with callbacks."""
        callback_called = []

        def test_callback(event, results):
            callback_called.append((event, results))

        trainer.train_agent(
            env=mock_env,
            total_timesteps=200,
            callbacks=[test_callback]
        )

        assert len(callback_called) == 1
        assert callback_called[0][0] == 'agent_trained'


# ============================================================================
# Combined Training Tests
# ============================================================================

class TestCombinedTraining:
    """Tests for combined training."""

    def test_train_combined_standard(self, trainer, mock_env):
        """Test standard combined training."""
        market_data = MockMarketData()

        results = trainer.train_combined(
            market_data=market_data,
            env=mock_env,
            predictor_epochs=2,
            agent_timesteps=200
        )

        assert results['predictor'] is not None
        assert results['agent'] is not None
        assert results['stages'] == []

    def test_train_combined_updates_history(self, trainer, mock_env):
        """Test that combined training updates history."""
        market_data = MockMarketData()

        trainer.train_combined(
            market_data=market_data,
            env=mock_env,
            predictor_epochs=2,
            agent_timesteps=200
        )

        assert len(trainer.training_history['combined']) == 1
        history_entry = trainer.training_history['combined'][0]
        assert 'timestamp' in history_entry
        assert 'results_summary' in history_entry

    def test_train_combined_with_curriculum(self, trainer, mock_env):
        """Test combined training with curriculum stages."""
        market_data = MockMarketData()

        curriculum_stages = [
            {'name': 'stage_1', 'difficulty': 0.5},
            {'name': 'stage_2', 'difficulty': 1.0}
        ]

        results = trainer.train_combined(
            market_data=market_data,
            env=mock_env,
            curriculum_stages=curriculum_stages
        )

        assert len(results['stages']) == 2
        assert results['stages'][0]['stage'] == 'stage_1'
        assert results['stages'][1]['stage'] == 'stage_2'


# ============================================================================
# Evaluation Tests
# ============================================================================

class TestEvaluation:
    """Tests for model evaluation."""

    def test_evaluate_predictor(self, trainer, mock_env):
        """Test predictor evaluation."""
        test_data = MockMarketData(n_bars=200)

        results = trainer.evaluate(
            test_data=test_data,
            env=None,
            n_eval_episodes=0
        )

        assert 'predictor' in results
        assert 'mae' in results['predictor']
        assert 'samples' in results['predictor']

    def test_evaluate_agent(self, trainer, mock_env):
        """Test agent evaluation."""
        results = trainer.evaluate(
            test_data=None,
            env=mock_env,
            n_eval_episodes=5
        )

        assert 'agent' in results
        assert 'avg_reward' in results['agent']
        assert results['agent']['episodes'] == 5

    def test_evaluate_both(self, trainer, mock_env):
        """Test evaluating both models."""
        test_data = MockMarketData(n_bars=200)

        results = trainer.evaluate(
            test_data=test_data,
            env=mock_env,
            n_eval_episodes=5
        )

        assert 'predictor' in results
        assert 'agent' in results


# ============================================================================
# Save/Load Tests
# ============================================================================

class TestSaveLoad:
    """Tests for model saving and loading."""

    def test_save_all(self, trainer, training_data, mock_env):
        """Test saving all models."""
        X_train, y_train, X_val, y_val = training_data

        # Train first
        trainer.train_predictor(X_train, y_train, X_val, y_val, epochs=2)
        trainer.train_agent(mock_env, total_timesteps=200)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_all(tmpdir)

            assert os.path.exists(os.path.join(tmpdir, 'predictor.pt'))
            assert os.path.exists(os.path.join(tmpdir, 'agent.pt'))
            assert os.path.exists(os.path.join(tmpdir, 'training_history.json'))

    def test_load_all(self, trainer, training_data, mock_env):
        """Test loading all models."""
        X_train, y_train, X_val, y_val = training_data

        # Train and save
        trainer.train_predictor(X_train, y_train, X_val, y_val, epochs=2)
        trainer.train_agent(mock_env, total_timesteps=200)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_all(tmpdir)

            # Load
            trainer.load_all(tmpdir)

            assert trainer.predictor._loaded_path == os.path.join(tmpdir, 'predictor.pt')
            assert trainer.agent._loaded_path == os.path.join(tmpdir, 'agent.pt')

    def test_load_restores_history(self, trainer, training_data, mock_env):
        """Test that loading restores training history."""
        X_train, y_train, X_val, y_val = training_data

        # Train and save
        trainer.train_predictor(X_train, y_train, X_val, y_val, epochs=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_all(tmpdir)

            # Clear history
            trainer.training_history = {'predictor': [], 'agent': [], 'combined': []}

            # Load
            trainer.load_all(tmpdir)

            assert len(trainer.training_history['predictor']) == 1


# ============================================================================
# Training Summary Tests
# ============================================================================

class TestTrainingSummary:
    """Tests for training summary."""

    def test_get_training_summary_empty(self, trainer):
        """Test summary with no training."""
        summary = trainer.get_training_summary()

        assert summary['predictor_runs'] == 0
        assert summary['agent_runs'] == 0
        assert summary['combined_runs'] == 0
        assert summary['latest_predictor'] is None
        assert summary['latest_agent'] is None

    def test_get_training_summary_after_training(self, trainer, training_data, mock_env):
        """Test summary after training."""
        X_train, y_train, X_val, y_val = training_data

        trainer.train_predictor(X_train, y_train, X_val, y_val, epochs=2)
        trainer.train_agent(mock_env, total_timesteps=200)

        summary = trainer.get_training_summary()

        assert summary['predictor_runs'] == 1
        assert summary['agent_runs'] == 1
        assert summary['latest_predictor'] is not None
        assert summary['latest_agent'] is not None


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_train_predictor_empty_data(self, trainer):
        """Test training with empty data."""
        X_train = np.array([]).reshape(0, 60, 10).astype(np.float32)
        y_train = np.array([]).astype(np.float32)
        X_val = np.array([]).reshape(0, 60, 10).astype(np.float32)
        y_val = np.array([]).astype(np.float32)

        # Should still return results (mock doesn't validate data)
        results = trainer.train_predictor(X_train, y_train, X_val, y_val, epochs=1)
        assert 'train_losses' in results

    def test_train_predictor_single_sample(self, trainer):
        """Test training with single sample."""
        X_train = np.random.randn(1, 60, 10).astype(np.float32)
        y_train = np.array([0.5]).astype(np.float32)
        X_val = np.random.randn(1, 60, 10).astype(np.float32)
        y_val = np.array([0.3]).astype(np.float32)

        results = trainer.train_predictor(X_train, y_train, X_val, y_val, epochs=1)
        assert 'train_losses' in results

    def test_train_agent_minimal_timesteps(self, trainer, mock_env):
        """Test training with minimal timesteps."""
        results = trainer.train_agent(env=mock_env, total_timesteps=10)
        assert 'episode_rewards' in results

    def test_multiple_training_runs(self, trainer, training_data, mock_env):
        """Test multiple consecutive training runs."""
        X_train, y_train, X_val, y_val = training_data

        # Run training multiple times
        for i in range(3):
            trainer.train_predictor(X_train, y_train, X_val, y_val, epochs=1)
            trainer.train_agent(mock_env, total_timesteps=100)

        assert len(trainer.training_history['predictor']) == 3
        assert len(trainer.training_history['agent']) == 3


# ============================================================================
# MLflow Integration Tests
# ============================================================================

class TestMLflowIntegration:
    """Tests for MLflow integration."""

    def test_train_predictor_with_mlflow_tracker(self, mock_predictor, mock_agent, mock_data_pipeline, training_data):
        """Test predictor training with MLflow tracker."""
        mock_tracker = Mock()
        mock_tracker.is_enabled = True
        mock_tracker.log_metrics = Mock()
        mock_tracker.log_training_summary = Mock()
        mock_tracker.log_predictor_model = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(
                predictor=mock_predictor,
                agent=mock_agent,
                data_pipeline=mock_data_pipeline,
                config={'checkpoint_dir': tmpdir},
                mlflow_tracker=mock_tracker
            )

            X_train, y_train, X_val, y_val = training_data
            trainer.train_predictor(X_train, y_train, X_val, y_val, epochs=2)

            # Verify MLflow methods were called
            assert mock_tracker.log_training_summary.called

    def test_train_agent_with_mlflow_tracker(self, mock_predictor, mock_agent, mock_data_pipeline, mock_env):
        """Test agent training with MLflow tracker."""
        mock_tracker = Mock()
        mock_tracker.is_enabled = True
        mock_tracker.log_metrics = Mock()
        mock_tracker.log_training_summary = Mock()
        mock_tracker.log_agent_model = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(
                predictor=mock_predictor,
                agent=mock_agent,
                data_pipeline=mock_data_pipeline,
                config={'checkpoint_dir': tmpdir},
                mlflow_tracker=mock_tracker
            )

            trainer.train_agent(mock_env, total_timesteps=200)

            # Verify MLflow methods were called
            assert mock_tracker.log_training_summary.called

    def test_train_without_mlflow_tracker(self, trainer, training_data):
        """Test that training works without MLflow tracker."""
        X_train, y_train, X_val, y_val = training_data

        # Should not raise any errors
        results = trainer.train_predictor(X_train, y_train, X_val, y_val, epochs=2)
        assert 'train_losses' in results


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
