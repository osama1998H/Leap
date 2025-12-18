"""Models module for Leap Trading System.

This module provides AI models for price prediction and trading decisions.

Protocols:
    - PredictorModel: Interface for prediction models (supervised learning)
    - AgentModel: Interface for RL agents

Factory Functions:
    - create_predictor(): Create predictor by type
    - create_agent(): Create agent by type
    - load_predictor(): Auto-load predictor from checkpoint
    - load_agent(): Auto-load agent from checkpoint

Built-in Implementations:
    - TransformerPredictor: Temporal Fusion Transformer for price prediction
    - PPOAgent: Proximal Policy Optimization for trading decisions

See ADR-0014 for extensibility design.
"""

# Protocols (interfaces)
from .base import PredictorModel, AgentModel

# Factory functions
from .factory import (
    create_predictor,
    create_agent,
    load_predictor,
    load_agent,
    list_predictors,
    list_agents,
    register_predictor,
    register_agent,
    get_predictor_class,
    get_agent_class,
)

# Concrete implementations (backward compatibility)
from .transformer import TemporalFusionTransformer, TransformerPredictor
from .ppo_agent import PPOAgent, ActorCritic

__all__ = [
    # Protocols
    'PredictorModel',
    'AgentModel',
    # Factory functions
    'create_predictor',
    'create_agent',
    'load_predictor',
    'load_agent',
    'list_predictors',
    'list_agents',
    'register_predictor',
    'register_agent',
    'get_predictor_class',
    'get_agent_class',
    # Concrete implementations
    'TemporalFusionTransformer',
    'TransformerPredictor',
    'PPOAgent',
    'ActorCritic',
]
