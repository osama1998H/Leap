"""
Leap Trading System - Model Factory

Registry and factory functions for creating and loading models.
Follows the BrokerGateway Protocol pattern from ADR-0010.

See ADR-0014 for design rationale.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from utils.checkpoint import load_checkpoint, CHECKPOINT_KEYS
from utils.device import resolve_device

from .base import PredictorModel, AgentModel

logger = logging.getLogger(__name__)


# =============================================================================
# Global Registries
# =============================================================================

_PREDICTOR_REGISTRY: Dict[str, Type] = {}
_AGENT_REGISTRY: Dict[str, Type] = {}


# =============================================================================
# Registration Decorators
# =============================================================================

def register_predictor(name: str):
    """
    Decorator to register a predictor model class.

    Args:
        name: Unique identifier for the model type (e.g., 'transformer', 'lstm')

    Example:
        @register_predictor('lstm')
        class LSTMPredictor:
            def __init__(self, input_dim: int, config: dict = None, device: str = 'auto'):
                ...
    """
    def decorator(cls: Type) -> Type:
        if name in _PREDICTOR_REGISTRY:
            logger.warning(f"Predictor '{name}' already registered, overwriting")
        _PREDICTOR_REGISTRY[name] = cls
        logger.debug(f"Registered predictor: {name}")
        return cls
    return decorator


def register_agent(name: str):
    """
    Decorator to register an agent model class.

    Args:
        name: Unique identifier for the model type (e.g., 'ppo', 'a2c')

    Example:
        @register_agent('a2c')
        class A2CAgent:
            def __init__(self, state_dim: int, action_dim: int = 4, config: dict = None, device: str = 'auto'):
                ...
    """
    def decorator(cls: Type) -> Type:
        if name in _AGENT_REGISTRY:
            logger.warning(f"Agent '{name}' already registered, overwriting")
        _AGENT_REGISTRY[name] = cls
        logger.debug(f"Registered agent: {name}")
        return cls
    return decorator


# =============================================================================
# Factory Functions
# =============================================================================

def create_predictor(
    model_type: str = 'transformer',
    input_dim: int = None,
    config: Optional[Dict[str, Any]] = None,
    device: str = 'auto'
) -> PredictorModel:
    """
    Factory function to create predictor models.

    Args:
        model_type: Type of predictor ('transformer', etc.)
        input_dim: Input feature dimension (required)
        config: Model configuration dictionary
        device: Device for model ('auto', 'cuda', 'mps', 'cpu')

    Returns:
        PredictorModel instance

    Raises:
        ValueError: If model_type is not registered or input_dim is missing

    Example:
        >>> predictor = create_predictor('transformer', input_dim=128, config={'d_model': 256})
    """
    if input_dim is None:
        raise ValueError("input_dim is required for predictor creation")

    if model_type not in _PREDICTOR_REGISTRY:
        available = list_predictors()
        raise ValueError(
            f"Unknown predictor type: '{model_type}'. "
            f"Available types: {available}"
        )

    cls = _PREDICTOR_REGISTRY[model_type]
    logger.info(f"Creating predictor: {model_type} (input_dim={input_dim})")

    return cls(input_dim=input_dim, config=config, device=device)


def create_agent(
    model_type: str = 'ppo',
    state_dim: int = None,
    action_dim: int = 4,
    config: Optional[Dict[str, Any]] = None,
    device: str = 'auto'
) -> AgentModel:
    """
    Factory function to create agent models.

    Args:
        model_type: Type of agent ('ppo', etc.)
        state_dim: State/observation dimension (required)
        action_dim: Action space dimension (default: 4 for HOLD/BUY/SELL/CLOSE)
        config: Agent configuration dictionary
        device: Device for model ('auto', 'cuda', 'mps', 'cpu')

    Returns:
        AgentModel instance

    Raises:
        ValueError: If model_type is not registered or state_dim is missing

    Example:
        >>> agent = create_agent('ppo', state_dim=72, action_dim=4, config={'learning_rate': 3e-4})
    """
    if state_dim is None:
        raise ValueError("state_dim is required for agent creation")

    if model_type not in _AGENT_REGISTRY:
        available = list_agents()
        raise ValueError(
            f"Unknown agent type: '{model_type}'. "
            f"Available types: {available}"
        )

    cls = _AGENT_REGISTRY[model_type]
    logger.info(f"Creating agent: {model_type} (state_dim={state_dim}, action_dim={action_dim})")

    return cls(state_dim=state_dim, action_dim=action_dim, config=config, device=device)


# =============================================================================
# Auto-Load Functions
# =============================================================================

def load_predictor(path: str, device: str = 'auto') -> PredictorModel:
    """
    Load a predictor from checkpoint with automatic type detection.

    Reads model_type from checkpoint metadata and instantiates the correct class.

    Args:
        path: Path to checkpoint file
        device: Device for model

    Returns:
        Loaded PredictorModel instance

    Raises:
        ValueError: If model_type not found in metadata or not registered

    Example:
        >>> predictor = load_predictor('saved_models/predictor.pt')
    """
    checkpoint = load_checkpoint(path, resolve_device(device))

    # Extract metadata
    metadata = checkpoint.get(CHECKPOINT_KEYS['METADATA'], {})
    if isinstance(metadata, dict):
        model_type = metadata.get('model_type', '')
        input_dim = metadata.get('input_dim')
    else:
        # Handle CheckpointMetadata dataclass
        model_type = getattr(metadata, 'model_type', '')
        input_dim = getattr(metadata, 'input_dim', None)

    if not model_type:
        raise ValueError(
            f"Cannot determine model_type from checkpoint at '{path}'. "
            f"Metadata: {metadata}"
        )

    if input_dim is None:
        raise ValueError(
            f"Cannot determine input_dim from checkpoint at '{path}'. "
            f"Metadata: {metadata}"
        )

    if model_type not in _PREDICTOR_REGISTRY:
        available = list_predictors()
        raise ValueError(
            f"Unknown predictor type in checkpoint: '{model_type}'. "
            f"Available types: {available}"
        )

    # Get config from checkpoint
    config = checkpoint.get(CHECKPOINT_KEYS['CONFIG'], {})

    # Create model and load weights
    predictor = create_predictor(
        model_type=model_type,
        input_dim=input_dim,
        config=config,
        device=device
    )
    predictor.load(path)

    logger.info(f"Loaded predictor from {path} (type={model_type}, input_dim={input_dim})")
    return predictor


def load_agent(path: str, device: str = 'auto') -> AgentModel:
    """
    Load an agent from checkpoint with automatic type detection.

    Reads model_type from checkpoint metadata and instantiates the correct class.

    Args:
        path: Path to checkpoint file
        device: Device for model

    Returns:
        Loaded AgentModel instance

    Raises:
        ValueError: If model_type not found in metadata or not registered

    Example:
        >>> agent = load_agent('saved_models/agent.pt')
    """
    checkpoint = load_checkpoint(path, resolve_device(device))

    # Extract metadata
    metadata = checkpoint.get(CHECKPOINT_KEYS['METADATA'], {})
    if isinstance(metadata, dict):
        model_type = metadata.get('model_type', '')
        state_dim = metadata.get('state_dim')
        action_dim = metadata.get('action_dim', 4)
    else:
        # Handle CheckpointMetadata dataclass
        model_type = getattr(metadata, 'model_type', '')
        state_dim = getattr(metadata, 'state_dim', None)
        action_dim = getattr(metadata, 'action_dim', 4)

    if not model_type:
        raise ValueError(
            f"Cannot determine model_type from checkpoint at '{path}'. "
            f"Metadata: {metadata}"
        )

    if state_dim is None:
        raise ValueError(
            f"Cannot determine state_dim from checkpoint at '{path}'. "
            f"Metadata: {metadata}"
        )

    if model_type not in _AGENT_REGISTRY:
        available = list_agents()
        raise ValueError(
            f"Unknown agent type in checkpoint: '{model_type}'. "
            f"Available types: {available}"
        )

    # Get config from checkpoint
    config = checkpoint.get(CHECKPOINT_KEYS['CONFIG'], {})

    # Create model and load weights
    agent = create_agent(
        model_type=model_type,
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        device=device
    )
    agent.load(path)

    logger.info(f"Loaded agent from {path} (type={model_type}, state_dim={state_dim})")
    return agent


# =============================================================================
# Discovery Functions
# =============================================================================

def list_predictors() -> List[str]:
    """
    List all registered predictor types.

    Returns:
        List of predictor type names

    Example:
        >>> list_predictors()
        ['transformer']
    """
    return list(_PREDICTOR_REGISTRY.keys())


def list_agents() -> List[str]:
    """
    List all registered agent types.

    Returns:
        List of agent type names

    Example:
        >>> list_agents()
        ['ppo']
    """
    return list(_AGENT_REGISTRY.keys())


def get_predictor_class(model_type: str) -> Type:
    """
    Get predictor class by name.

    Args:
        model_type: Predictor type name

    Returns:
        Predictor class

    Raises:
        ValueError: If model_type not registered
    """
    if model_type not in _PREDICTOR_REGISTRY:
        available = list_predictors()
        raise ValueError(
            f"Unknown predictor type: '{model_type}'. "
            f"Available types: {available}"
        )
    return _PREDICTOR_REGISTRY[model_type]


def get_agent_class(model_type: str) -> Type:
    """
    Get agent class by name.

    Args:
        model_type: Agent type name

    Returns:
        Agent class

    Raises:
        ValueError: If model_type not registered
    """
    if model_type not in _AGENT_REGISTRY:
        available = list_agents()
        raise ValueError(
            f"Unknown agent type: '{model_type}'. "
            f"Available types: {available}"
        )
    return _AGENT_REGISTRY[model_type]


# =============================================================================
# Built-in Model Registration
# =============================================================================

def _register_builtin_models():
    """
    Register built-in model implementations.

    Called automatically on module import.
    """
    # Import here to avoid circular imports
    from .transformer import TransformerPredictor
    from .ppo_agent import PPOAgent

    # Register if not already registered
    if 'transformer' not in _PREDICTOR_REGISTRY:
        register_predictor('transformer')(TransformerPredictor)

    if 'ppo' not in _AGENT_REGISTRY:
        register_agent('ppo')(PPOAgent)

    logger.debug(f"Built-in models registered: predictors={list_predictors()}, agents={list_agents()}")


# Auto-register on module import
_register_builtin_models()
