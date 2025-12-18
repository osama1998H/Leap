"""
Tests for model factory and protocols.

Tests the model extensibility system introduced in ADR-0014.
"""

import pytest
import numpy as np
import tempfile
import os

from models import (
    # Protocols
    PredictorModel,
    AgentModel,
    # Factory functions
    create_predictor,
    create_agent,
    list_predictors,
    list_agents,
    register_predictor,
    register_agent,
    get_predictor_class,
    get_agent_class,
    load_predictor,
    load_agent,
    # Concrete implementations
    TransformerPredictor,
    PPOAgent,
)


class TestProtocolCompliance:
    """Test that built-in models satisfy protocols."""

    def test_transformer_is_predictor_model(self):
        """TransformerPredictor should be a PredictorModel."""
        predictor = create_predictor('transformer', input_dim=10)
        assert isinstance(predictor, PredictorModel)

    def test_ppo_is_agent_model(self):
        """PPOAgent should be an AgentModel."""
        agent = create_agent('ppo', state_dim=10)
        assert isinstance(agent, AgentModel)

    def test_transformer_has_required_attributes(self):
        """TransformerPredictor should have required protocol attributes."""
        predictor = create_predictor('transformer', input_dim=10)
        assert hasattr(predictor, 'input_dim')
        assert hasattr(predictor, 'config')
        assert hasattr(predictor, 'device')
        assert predictor.input_dim == 10

    def test_ppo_has_required_attributes(self):
        """PPOAgent should have required protocol attributes."""
        agent = create_agent('ppo', state_dim=10, action_dim=4)
        assert hasattr(agent, 'state_dim')
        assert hasattr(agent, 'action_dim')
        assert hasattr(agent, 'config')
        assert hasattr(agent, 'device')
        assert agent.state_dim == 10
        assert agent.action_dim == 4


class TestFactory:
    """Test factory functions."""

    def test_create_predictor_transformer(self):
        """create_predictor should create TransformerPredictor."""
        predictor = create_predictor('transformer', input_dim=10)
        assert predictor.input_dim == 10
        assert isinstance(predictor, TransformerPredictor)

    def test_create_predictor_with_config(self):
        """create_predictor should accept config dict."""
        predictor = create_predictor(
            'transformer',
            input_dim=10,
            config={'d_model': 64, 'n_heads': 4}
        )
        assert predictor.d_model == 64
        assert predictor.n_heads == 4

    def test_create_agent_ppo(self):
        """create_agent should create PPOAgent."""
        agent = create_agent('ppo', state_dim=10, action_dim=4)
        assert agent.state_dim == 10
        assert agent.action_dim == 4
        assert isinstance(agent, PPOAgent)

    def test_create_agent_default_action_dim(self):
        """create_agent should default to 4 actions."""
        agent = create_agent('ppo', state_dim=10)
        assert agent.action_dim == 4

    def test_create_predictor_invalid_type(self):
        """create_predictor should raise for unknown type."""
        with pytest.raises(ValueError, match="Unknown predictor type"):
            create_predictor('invalid', input_dim=10)

    def test_create_agent_invalid_type(self):
        """create_agent should raise for unknown type."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent('invalid', state_dim=10)

    def test_create_predictor_missing_input_dim(self):
        """create_predictor should raise if input_dim missing."""
        with pytest.raises(ValueError, match="input_dim is required"):
            create_predictor('transformer')

    def test_create_agent_missing_state_dim(self):
        """create_agent should raise if state_dim missing."""
        with pytest.raises(ValueError, match="state_dim is required"):
            create_agent('ppo')


class TestRegistry:
    """Test registration mechanism."""

    def test_list_predictors(self):
        """list_predictors should return registered types."""
        predictors = list_predictors()
        assert 'transformer' in predictors

    def test_list_agents(self):
        """list_agents should return registered types."""
        agents = list_agents()
        assert 'ppo' in agents

    def test_get_predictor_class(self):
        """get_predictor_class should return class by name."""
        cls = get_predictor_class('transformer')
        assert cls == TransformerPredictor

    def test_get_agent_class(self):
        """get_agent_class should return class by name."""
        cls = get_agent_class('ppo')
        assert cls == PPOAgent

    def test_get_predictor_class_invalid(self):
        """get_predictor_class should raise for unknown type."""
        with pytest.raises(ValueError, match="Unknown predictor type"):
            get_predictor_class('invalid')

    def test_get_agent_class_invalid(self):
        """get_agent_class should raise for unknown type."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            get_agent_class('invalid')


class TestCustomRegistration:
    """Test registering custom models."""

    def test_register_custom_predictor(self):
        """Custom predictor can be registered and used."""

        @register_predictor('test_predictor')
        class TestPredictor:
            def __init__(self, input_dim, config=None, device='auto'):
                self.input_dim = input_dim
                self.config = config or {}
                self.device = device

        assert 'test_predictor' in list_predictors()

        predictor = create_predictor('test_predictor', input_dim=20)
        assert predictor.input_dim == 20

    def test_register_custom_agent(self):
        """Custom agent can be registered and used."""

        @register_agent('test_agent')
        class TestAgent:
            def __init__(self, state_dim, action_dim=4, config=None, device='auto'):
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.config = config or {}
                self.device = device

        assert 'test_agent' in list_agents()

        agent = create_agent('test_agent', state_dim=20)
        assert agent.state_dim == 20


class TestModelOperations:
    """Test model training and inference operations."""

    def test_predictor_predict_shape(self):
        """Predictor predict should return correct shape."""
        predictor = create_predictor('transformer', input_dim=10)

        # Create dummy input: (batch, seq_len, features)
        X = np.random.randn(2, 60, 10).astype(np.float32)
        result = predictor.predict(X)

        assert 'prediction' in result
        assert result['prediction'].shape[0] == 2  # batch size

    def test_agent_select_action(self):
        """Agent select_action should return action, log_prob, value."""
        agent = create_agent('ppo', state_dim=10)

        state = np.random.randn(10).astype(np.float32)
        action, log_prob, value = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action <= 3  # HOLD, BUY, SELL, CLOSE
        assert isinstance(log_prob, float)
        assert isinstance(value, float)


class TestCheckpointAutoload:
    """Test checkpoint auto-loading with model type detection."""

    def test_save_load_predictor(self):
        """Predictor can be saved and auto-loaded."""
        predictor = create_predictor('transformer', input_dim=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'predictor.pt')
            predictor.save(path)

            # Load with auto-detection
            loaded = load_predictor(path)

            assert isinstance(loaded, TransformerPredictor)
            assert loaded.input_dim == 10

    def test_save_load_agent(self):
        """Agent can be saved and auto-loaded."""
        agent = create_agent('ppo', state_dim=10, action_dim=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'agent.pt')
            agent.save(path)

            # Load with auto-detection
            loaded = load_agent(path)

            assert isinstance(loaded, PPOAgent)
            assert loaded.state_dim == 10
            assert loaded.action_dim == 4


class TestBackwardCompatibility:
    """Test backward compatibility with direct class usage."""

    def test_direct_transformer_creation(self):
        """TransformerPredictor can still be created directly."""
        predictor = TransformerPredictor(input_dim=10)
        assert predictor.input_dim == 10

    def test_direct_ppo_creation(self):
        """PPOAgent can still be created directly."""
        agent = PPOAgent(state_dim=10, action_dim=4)
        assert agent.state_dim == 10
        assert agent.action_dim == 4
