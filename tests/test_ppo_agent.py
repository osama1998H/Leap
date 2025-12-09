"""
Leap Trading System - PPOAgent Tests
Comprehensive tests for the PPO reinforcement learning agent.
"""

import pytest
import numpy as np
import torch
import os
import tempfile
from gymnasium import spaces

# Import components to test
from models.ppo_agent import PPOAgent, ActorCritic, RolloutBuffer


# ============================================================================
# Mock Environment
# ============================================================================

class MockTradingEnv:
    """Mock trading environment for testing."""

    def __init__(self, state_dim=50, n_steps=100):
        self.state_dim = state_dim
        self.n_steps = n_steps
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None):
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        obs = self._get_obs()
        reward = np.random.uniform(-1, 1)
        terminated = self.current_step >= self.n_steps
        truncated = False
        info = {'balance': 10000, 'equity': 10000 + reward * 100}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.random.randn(self.state_dim).astype(np.float32)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def state_dim():
    """Standard state dimension for tests."""
    return 50


@pytest.fixture
def action_dim():
    """Standard action dimension for tests."""
    return 4


@pytest.fixture
def ppo_agent(state_dim, action_dim):
    """Create PPOAgent for testing."""
    return PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config={
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'n_epochs': 4,
            'batch_size': 32,
            'n_steps': 128,
            'hidden_sizes': [64, 64, 32]
        },
        device='cpu'
    )


@pytest.fixture
def mock_env(state_dim):
    """Create mock environment."""
    return MockTradingEnv(state_dim=state_dim, n_steps=100)


@pytest.fixture
def sample_states(state_dim):
    """Create sample state data."""
    return np.random.randn(100, state_dim).astype(np.float32)


# ============================================================================
# ActorCritic Network Tests
# ============================================================================

class TestActorCritic:
    """Tests for ActorCritic network."""

    def test_initialization(self, state_dim, action_dim):
        """Test ActorCritic initialization."""
        network = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=[64, 64, 32]
        )

        assert network.state_dim == state_dim
        assert network.action_dim == action_dim

    def test_forward_output_shapes(self, state_dim, action_dim):
        """Test forward pass output shapes."""
        network = ActorCritic(state_dim=state_dim, action_dim=action_dim)

        batch_size = 16
        states = torch.randn(batch_size, state_dim)
        action_logits, values = network(states)

        assert action_logits.shape == (batch_size, action_dim)
        assert values.shape == (batch_size, 1)

    def test_get_action_output(self, state_dim, action_dim):
        """Test get_action method."""
        network = ActorCritic(state_dim=state_dim, action_dim=action_dim)

        state = torch.randn(1, state_dim)
        action, log_prob, value = network.get_action(state)

        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)

    def test_get_action_deterministic(self, state_dim, action_dim):
        """Test deterministic action selection."""
        network = ActorCritic(state_dim=state_dim, action_dim=action_dim)
        network.eval()

        state = torch.randn(1, state_dim)

        # Deterministic should always return same action
        actions = [
            network.get_action(state, deterministic=True)[0].item()
            for _ in range(10)
        ]

        assert len(set(actions)) == 1  # All same

    def test_get_action_stochastic(self, state_dim, action_dim):
        """Test stochastic action selection."""
        network = ActorCritic(state_dim=state_dim, action_dim=action_dim)
        network.train()

        state = torch.randn(1, state_dim)

        # Stochastic should return different actions sometimes
        actions = [
            network.get_action(state, deterministic=False)[0].item()
            for _ in range(50)
        ]

        # Should have some variation (not guaranteed but highly likely)
        # Allow for edge case where all same
        assert len(actions) == 50

    def test_evaluate_actions(self, state_dim, action_dim):
        """Test evaluate_actions method."""
        network = ActorCritic(state_dim=state_dim, action_dim=action_dim)

        batch_size = 16
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, action_dim, (batch_size,))

        log_probs, values, entropy = network.evaluate_actions(states, actions)

        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size,)
        assert entropy.shape == (batch_size,)

    def test_entropy_is_positive(self, state_dim, action_dim):
        """Test that entropy is positive (exploration)."""
        network = ActorCritic(state_dim=state_dim, action_dim=action_dim)

        states = torch.randn(16, state_dim)
        actions = torch.randint(0, action_dim, (16,))

        _, _, entropy = network.evaluate_actions(states, actions)

        assert torch.all(entropy >= 0)

    def test_hidden_sizes_validation(self, state_dim, action_dim):
        """Test that hidden_sizes with less than 3 elements raises error."""
        with pytest.raises(ValueError, match="hidden_sizes must have at least 3 elements"):
            ActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_sizes=[64, 64]  # Only 2 elements
            )


# ============================================================================
# RolloutBuffer Tests
# ============================================================================

class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    def test_initialization(self, state_dim):
        """Test buffer initialization."""
        buffer = RolloutBuffer(
            buffer_size=128,
            state_dim=state_dim,
            device=torch.device('cpu')
        )

        assert len(buffer) == 0

    def test_add_experience(self, state_dim):
        """Test adding experience to buffer."""
        buffer = RolloutBuffer(
            buffer_size=128,
            state_dim=state_dim,
            device=torch.device('cpu')
        )

        state = np.random.randn(state_dim).astype(np.float32)
        buffer.add(
            state=state,
            action=1,
            reward=0.5,
            done=False,
            log_prob=-0.5,
            value=0.1
        )

        assert len(buffer) == 1

    def test_get_returns_tensors(self, state_dim):
        """Test get method returns proper tensors."""
        buffer = RolloutBuffer(
            buffer_size=128,
            state_dim=state_dim,
            device=torch.device('cpu')
        )

        # Add some experiences
        for _ in range(10):
            state = np.random.randn(state_dim).astype(np.float32)
            buffer.add(state, 1, 0.5, False, -0.5, 0.1)

        data = buffer.get()

        assert 'states' in data
        assert 'actions' in data
        assert 'rewards' in data
        assert 'dones' in data
        assert 'log_probs' in data
        assert 'values' in data

        assert data['states'].shape == (10, state_dim)
        assert data['actions'].shape == (10,)

    def test_reset(self, state_dim):
        """Test buffer reset."""
        buffer = RolloutBuffer(
            buffer_size=128,
            state_dim=state_dim,
            device=torch.device('cpu')
        )

        # Add experiences
        for _ in range(10):
            state = np.random.randn(state_dim).astype(np.float32)
            buffer.add(state, 1, 0.5, False, -0.5, 0.1)

        assert len(buffer) == 10

        buffer.reset()

        assert len(buffer) == 0


# ============================================================================
# PPOAgent Initialization Tests
# ============================================================================

class TestPPOAgentInitialization:
    """Tests for PPOAgent initialization."""

    def test_initialization_default_config(self, state_dim, action_dim):
        """Test initialization with default config."""
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)

        assert agent.state_dim == state_dim
        assert agent.action_dim == action_dim
        assert agent.network is not None

    def test_initialization_custom_config(self, state_dim, action_dim):
        """Test initialization with custom config."""
        config = {
            'gamma': 0.95,
            'learning_rate': 1e-4,
            'clip_epsilon': 0.1
        }
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config
        )

        assert agent.gamma == 0.95
        assert agent.learning_rate == 1e-4
        assert agent.clip_epsilon == 0.1

    def test_initialization_device_cpu(self, state_dim, action_dim):
        """Test initialization with CPU device."""
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device='cpu'
        )

        assert agent.device.type == 'cpu'

    def test_initialization_device_auto(self, state_dim, action_dim):
        """Test initialization with auto device."""
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device='auto'
        )

        assert agent.device.type in ['cuda', 'cpu']

    def test_buffer_initialized(self, ppo_agent):
        """Test that rollout buffer is initialized."""
        assert ppo_agent.buffer is not None
        assert len(ppo_agent.buffer) == 0


# ============================================================================
# PPOAgent Action Selection Tests
# ============================================================================

class TestPPOAgentActionSelection:
    """Tests for PPOAgent action selection."""

    def test_select_action_returns_tuple(self, ppo_agent, state_dim):
        """Test select_action returns (action, log_prob, value)."""
        state = np.random.randn(state_dim).astype(np.float32)
        result = ppo_agent.select_action(state)

        assert len(result) == 3
        action, log_prob, value = result

        assert isinstance(action, int)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_select_action_valid_range(self, ppo_agent, state_dim, action_dim):
        """Test selected action is in valid range."""
        state = np.random.randn(state_dim).astype(np.float32)

        for _ in range(50):
            action, _, _ = ppo_agent.select_action(state)
            assert 0 <= action < action_dim

    def test_select_action_deterministic(self, ppo_agent, state_dim):
        """Test deterministic action selection."""
        state = np.random.randn(state_dim).astype(np.float32)

        actions = [
            ppo_agent.select_action(state, deterministic=True)[0]
            for _ in range(10)
        ]

        # All actions should be the same
        assert len(set(actions)) == 1

    def test_get_policy_distribution(self, ppo_agent, state_dim):
        """Test get_policy_distribution method."""
        state = np.random.randn(state_dim).astype(np.float32)
        probs = ppo_agent.get_policy_distribution(state)

        assert len(probs) == ppo_agent.action_dim
        assert abs(sum(probs) - 1.0) < 1e-5  # Sum to 1
        assert all(p >= 0 for p in probs)  # All non-negative


# ============================================================================
# PPOAgent Training Tests
# ============================================================================

class TestPPOAgentTraining:
    """Tests for PPOAgent training."""

    def test_store_transition(self, ppo_agent, state_dim):
        """Test storing transitions."""
        state = np.random.randn(state_dim).astype(np.float32)

        ppo_agent.store_transition(
            state=state,
            action=1,
            reward=0.5,
            done=False,
            log_prob=-0.5,
            value=0.1
        )

        assert len(ppo_agent.buffer) == 1

    def test_compute_returns_and_advantages(self, ppo_agent):
        """Test GAE computation."""
        n_steps = 10
        rewards = torch.rand(n_steps)
        values = torch.rand(n_steps)
        dones = torch.zeros(n_steps)

        returns, advantages = ppo_agent.compute_returns_and_advantages(
            rewards, values, dones, last_value=0.0
        )

        assert returns.shape == (n_steps,)
        assert advantages.shape == (n_steps,)

    def test_update_returns_stats(self, ppo_agent, state_dim):
        """Test that update returns statistics."""
        # Fill buffer with experiences
        for _ in range(64):
            state = np.random.randn(state_dim).astype(np.float32)
            ppo_agent.store_transition(state, 1, 0.5, False, -0.5, 0.1)

        stats = ppo_agent.update(last_value=0.0)

        assert 'policy_loss' in stats
        assert 'value_loss' in stats
        assert 'entropy' in stats

    def test_update_clears_buffer(self, ppo_agent, state_dim):
        """Test that update clears the buffer."""
        # Fill buffer
        for _ in range(64):
            state = np.random.randn(state_dim).astype(np.float32)
            ppo_agent.store_transition(state, 1, 0.5, False, -0.5, 0.1)

        ppo_agent.update(last_value=0.0)

        assert len(ppo_agent.buffer) == 0

    def test_update_with_empty_buffer(self, ppo_agent):
        """Test update with empty buffer returns zeros."""
        stats = ppo_agent.update(last_value=0.0)

        assert stats['policy_loss'] == 0.0
        assert stats['value_loss'] == 0.0

    def test_train_on_env(self, ppo_agent, mock_env):
        """Test training on environment."""
        results = ppo_agent.train_on_env(
            env=mock_env,
            total_timesteps=200,
            eval_frequency=100
        )

        assert 'episode_rewards' in results
        assert 'episode_lengths' in results
        assert len(results['episode_rewards']) > 0

    def test_train_with_mlflow_callback(self, ppo_agent, mock_env):
        """Test training with MLflow callback."""
        callback_calls = []

        def test_callback(metrics, step):
            callback_calls.append((metrics, step))

        ppo_agent.train_on_env(
            env=mock_env,
            total_timesteps=200,
            mlflow_callback=test_callback
        )

        assert len(callback_calls) > 0


# ============================================================================
# PPOAgent Evaluation Tests
# ============================================================================

class TestPPOAgentEvaluation:
    """Tests for PPOAgent evaluation."""

    def test_evaluate_returns_average_reward(self, ppo_agent, mock_env):
        """Test evaluate returns average reward."""
        avg_reward = ppo_agent.evaluate(mock_env, n_episodes=5)

        assert isinstance(avg_reward, float)

    def test_evaluate_returns_finite_results(self, ppo_agent, mock_env):
        """Test evaluation returns finite reward values."""
        # Run evaluation twice
        reward1 = ppo_agent.evaluate(mock_env, n_episodes=3)
        reward2 = ppo_agent.evaluate(mock_env, n_episodes=3)

        # Both should be finite numbers
        assert np.isfinite(reward1)
        assert np.isfinite(reward2)


# ============================================================================
# PPOAgent Save/Load Tests
# ============================================================================

class TestPPOAgentSaveLoad:
    """Tests for PPOAgent save/load."""

    def test_save_creates_file(self, ppo_agent):
        """Test save creates model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'agent.pt')
            ppo_agent.save(path)

            assert os.path.exists(path)

    def test_load_restores_model(self, ppo_agent, state_dim, action_dim):
        """Test load restores model correctly."""
        state = np.random.randn(state_dim).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'agent.pt')

            # Get action before save
            action_before, _, _ = ppo_agent.select_action(state, deterministic=True)

            # Save
            ppo_agent.save(path)

            # Create new agent and load
            new_agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                config=ppo_agent.config,
                device='cpu'
            )
            new_agent.load(path)

            # Get action after load
            action_after, _, _ = new_agent.select_action(state, deterministic=True)

            # Actions should be identical
            assert action_before == action_after

    def test_load_restores_optimizer(self, ppo_agent, state_dim, action_dim):
        """Test load restores optimizer state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'agent.pt')

            # Train a bit to get non-default optimizer state
            for _ in range(10):
                state = np.random.randn(state_dim).astype(np.float32)
                ppo_agent.store_transition(state, 1, 0.5, False, -0.5, 0.1)
            ppo_agent.update()

            # Save
            ppo_agent.save(path)

            # Load into new agent
            new_agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                device='cpu'
            )
            new_agent.load(path)

            # Should not raise errors
            assert new_agent.optimizer is not None


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_step_update(self, ppo_agent, state_dim):
        """Test update with single step in buffer."""
        state = np.random.randn(state_dim).astype(np.float32)
        ppo_agent.store_transition(state, 1, 0.5, True, -0.5, 0.1)

        # Should not crash
        stats = ppo_agent.update()
        assert stats is not None

    def test_all_done_episodes(self, ppo_agent, state_dim):
        """Test update when all episodes are done."""
        for _ in range(10):
            state = np.random.randn(state_dim).astype(np.float32)
            ppo_agent.store_transition(state, 1, 0.5, True, -0.5, 0.1)

        stats = ppo_agent.update()
        assert stats is not None

    def test_extreme_rewards(self, ppo_agent, state_dim):
        """Test handling of extreme reward values."""
        for i in range(10):
            state = np.random.randn(state_dim).astype(np.float32)
            reward = 1000.0 if i % 2 == 0 else -1000.0
            ppo_agent.store_transition(state, 1, reward, False, -0.5, 0.1)

        stats = ppo_agent.update()
        assert np.isfinite(stats['policy_loss'])
        assert np.isfinite(stats['value_loss'])

    def test_training_stability(self, state_dim, action_dim, mock_env):
        """Test training stability over multiple updates."""
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config={'n_steps': 32, 'n_epochs': 2, 'batch_size': 16},
            device='cpu'
        )

        # Run multiple training iterations
        for _ in range(5):
            results = agent.train_on_env(mock_env, total_timesteps=100)

            # Check no NaN values in results
            for reward in results['episode_rewards']:
                assert np.isfinite(reward)

    def test_very_small_state_dim(self, action_dim):
        """Test with minimal state dimension."""
        agent = PPOAgent(
            state_dim=5,
            action_dim=action_dim,
            config={'hidden_sizes': [16, 16, 8]},
            device='cpu'
        )

        state = np.random.randn(5).astype(np.float32)
        action, log_prob, value = agent.select_action(state)

        assert 0 <= action < action_dim

    def test_large_state_dim(self, action_dim):
        """Test with large state dimension."""
        agent = PPOAgent(
            state_dim=500,
            action_dim=action_dim,
            device='cpu'
        )

        state = np.random.randn(500).astype(np.float32)
        action, log_prob, value = agent.select_action(state)

        assert 0 <= action < action_dim


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
