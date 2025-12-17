"""
Leap Trading System - Online Learning Interface

MAJOR-7 fix: Unified interface for online learning across model types.

This module provides:
- OnlineLearningData: Unified data container for both supervised and RL models
- OnlineLearnable: Protocol/ABC for models that support online learning
- Adapter functions to bridge different model signatures
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class OnlineLearningData:
    """
    Unified data container for online learning updates.

    This dataclass provides a common interface for passing data to different
    model types during online learning:

    For supervised models (TransformerPredictor):
        - features: X_new (batch of input sequences)
        - targets: y_new (prediction targets)
        - learning_rate: Optional override for learning rate

    For RL models (PPOAgent):
        - states: Recent states from experience buffer
        - actions: Actions taken
        - rewards: Rewards received
        - dones: Episode termination flags
        - n_samples: Number of samples to use from buffer
        - n_epochs: Number of training epochs

    Common parameters:
        - learning_rate: Optional learning rate override
    """
    # Supervised learning data (for TransformerPredictor)
    features: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None

    # RL data (for PPOAgent)
    states: Optional[np.ndarray] = None
    actions: Optional[np.ndarray] = None
    rewards: Optional[np.ndarray] = None
    dones: Optional[np.ndarray] = None

    # Buffer sampling parameters (for PPOAgent)
    n_samples: int = 256
    n_epochs: int = 5

    # Common parameters
    learning_rate: Optional[float] = None

    def is_supervised_data(self) -> bool:
        """Check if this contains supervised learning data."""
        return self.features is not None and self.targets is not None

    def is_rl_data(self) -> bool:
        """Check if this contains RL data (or uses buffer sampling)."""
        return (
            self.states is not None or
            self.n_samples > 0  # PPO samples from internal buffer
        )


@dataclass
class OnlineLearningResult:
    """
    Result from an online learning update.

    Provides a unified structure for both model types.
    """
    success: bool = True
    loss: float = 0.0
    learning_rate: float = 0.0
    samples_used: int = 0
    extra_metrics: Dict[str, float] = field(default_factory=dict)


@runtime_checkable
class OnlineLearnable(Protocol):
    """
    Protocol for models that support online learning.

    Unifies the online_update() interface across model types.
    Models implementing this protocol must provide:
    - online_update(): Method to update model with new data
    - supports_online_learning(): Check if model is configured for online learning
    """

    def online_update(self, data: OnlineLearningData) -> OnlineLearningResult:
        """
        Update model with new data using unified interface.

        Args:
            data: OnlineLearningData container with appropriate fields populated

        Returns:
            OnlineLearningResult with loss/metric values
        """
        ...

    def supports_online_learning(self) -> bool:
        """Check if model is configured for online learning."""
        ...


class OnlineLearningAdapter:
    """
    Adapter to provide unified online learning interface for existing models.

    Wraps TransformerPredictor and PPOAgent to provide consistent interface
    without modifying their core implementations.
    """

    def __init__(self, model: Any, model_type: str = 'auto'):
        """
        Initialize adapter for a model.

        Args:
            model: The model to wrap (TransformerPredictor or PPOAgent)
            model_type: 'transformer', 'ppo', or 'auto' to detect
        """
        self.model = model

        # Detect model type
        if model_type == 'auto':
            self.model_type = self._detect_model_type(model)
        else:
            self.model_type = model_type

        logger.debug(f"OnlineLearningAdapter initialized for {self.model_type}")

    def _detect_model_type(self, model: Any) -> str:
        """Detect model type from class name."""
        class_name = model.__class__.__name__
        if 'Transformer' in class_name or 'Predictor' in class_name:
            return 'transformer'
        elif 'PPO' in class_name or 'Agent' in class_name:
            return 'ppo'
        else:
            raise ValueError(f"Unknown model type: {class_name}")

    def online_update(self, data: OnlineLearningData) -> OnlineLearningResult:
        """
        Perform online update using unified interface.

        Delegates to the appropriate model method based on model type.
        """
        if self.model_type == 'transformer':
            return self._update_transformer(data)
        elif self.model_type == 'ppo':
            return self._update_ppo(data)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _update_transformer(self, data: OnlineLearningData) -> OnlineLearningResult:
        """Update transformer model with supervised data."""
        if not data.is_supervised_data():
            return OnlineLearningResult(
                success=False,
                extra_metrics={'error': 'Transformer requires features and targets'}
            )

        try:
            # Call transformer's online_update with its expected signature
            loss = self.model.online_update(
                X_new=data.features,
                y_new=data.targets,
                learning_rate=data.learning_rate
            )

            current_lr = self.model.optimizer.param_groups[0]['lr']

            return OnlineLearningResult(
                success=True,
                loss=loss,
                learning_rate=current_lr,
                samples_used=len(data.features) if data.features is not None else 0
            )
        except Exception as e:
            logger.exception(f"Transformer online update failed: {e}")
            return OnlineLearningResult(
                success=False,
                extra_metrics={'error': str(e)}
            )

    def _update_ppo(self, data: OnlineLearningData) -> OnlineLearningResult:
        """Update PPO agent from experience buffer."""
        try:
            # Call PPO's online_update with its expected signature
            result = self.model.online_update(
                n_samples=data.n_samples,
                n_epochs=data.n_epochs
            )

            if result is None:
                return OnlineLearningResult(
                    success=False,
                    extra_metrics={'error': 'Not enough samples in buffer'}
                )

            current_lr = self.model.optimizer.param_groups[0]['lr']

            return OnlineLearningResult(
                success=True,
                loss=result.get('online_loss', 0.0),
                learning_rate=current_lr,
                samples_used=data.n_samples
            )
        except Exception as e:
            logger.exception(f"PPO online update failed: {e}")
            return OnlineLearningResult(
                success=False,
                extra_metrics={'error': str(e)}
            )

    def supports_online_learning(self) -> bool:
        """Check if model supports online learning."""
        return hasattr(self.model, 'online_update')

    def get_online_learning_config(self) -> Dict[str, Any]:
        """Get current online learning configuration."""
        config = {
            'model_type': self.model_type,
            'enabled': self.supports_online_learning(),
        }

        if hasattr(self.model, 'config'):
            model_config = self.model.config
            if isinstance(model_config, dict):
                config['online_learning_rate'] = model_config.get('online_learning_rate')

        return config


def create_online_learner(model: Any) -> OnlineLearningAdapter:
    """
    Factory function to create an online learning adapter for a model.

    Args:
        model: TransformerPredictor or PPOAgent instance

    Returns:
        OnlineLearningAdapter wrapping the model

    Example:
        >>> predictor = TransformerPredictor(input_dim=128)
        >>> learner = create_online_learner(predictor)
        >>> data = OnlineLearningData(features=X_new, targets=y_new)
        >>> result = learner.online_update(data)
    """
    return OnlineLearningAdapter(model)
