"""
Leap Trading System - Model Base Protocols

Defines abstract interfaces for prediction models and RL agents.
All model implementations should satisfy these protocols.

See ADR-0014 for design rationale.
"""

import torch
import numpy as np
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Predictor Model Protocol
# =============================================================================

@runtime_checkable
class PredictorModel(Protocol):
    """
    Protocol for prediction models (supervised learning).

    This protocol defines the interface that all prediction models must satisfy.
    Models are expected to predict future price movements based on historical data.

    Built-in implementations:
        - TransformerPredictor: Temporal Fusion Transformer for time series forecasting

    Example:
        >>> predictor = create_predictor('transformer', input_dim=128)
        >>> predictor.train(X_train, y_train, X_val, y_val, epochs=100)
        >>> predictions = predictor.predict(X_test)
    """

    # Required attributes
    input_dim: int
    config: Dict[str, Any]
    device: torch.device

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        verbose: bool = True,
        mlflow_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train the model on historical data.

        Args:
            X_train: Training features of shape (samples, sequence_length, features)
            y_train: Training targets of shape (samples,) or (samples, 1)
            X_val: Optional validation features
            y_val: Optional validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            verbose: Whether to show progress
            mlflow_callback: Optional callback for MLflow logging

        Returns:
            Dictionary containing training metrics
        """
        ...

    def predict(
        self,
        X: np.ndarray,
        return_uncertainty: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions on input data.

        Args:
            X: Input features of shape (samples, sequence_length, features)
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dictionary containing:
                - 'prediction': Point predictions
                - 'quantiles': Quantile predictions (if model supports)
                - 'uncertainty': Uncertainty estimates (if return_uncertainty=True)
        """
        ...

    def online_update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        learning_rate: Optional[float] = None
    ) -> float:
        """
        Update model with new data (online learning).

        Uses a smaller learning rate to avoid catastrophic forgetting.

        Args:
            X_new: New features
            y_new: New targets
            learning_rate: Optional learning rate override

        Returns:
            Loss value from the update
        """
        ...

    def save(self, path: str) -> None:
        """
        Save model checkpoint.

        Checkpoint should include:
            - model_state_dict: Model weights
            - optimizer_state_dict: Optimizer state
            - config: Model configuration
            - training_history: Training metrics history
            - metadata: CheckpointMetadata with model_type

        Args:
            path: Path to save checkpoint
        """
        ...

    def load(self, path: str) -> None:
        """
        Load model checkpoint.

        Note: Model must be initialized with matching dimensions before loading.

        Args:
            path: Path to load checkpoint from
        """
        ...


# =============================================================================
# Agent Model Protocol
# =============================================================================

@runtime_checkable
class AgentModel(Protocol):
    """
    Protocol for reinforcement learning agents.

    This protocol defines the interface that all RL agents must satisfy.
    Agents make trading decisions based on market observations.

    Built-in implementations:
        - PPOAgent: Proximal Policy Optimization for discrete trading actions

    Action space (standard for trading):
        - 0: HOLD - No action
        - 1: BUY - Open long position
        - 2: SELL - Open short position
        - 3: CLOSE - Close all positions

    Example:
        >>> agent = create_agent('ppo', state_dim=72, action_dim=4)
        >>> agent.train_on_env(env, total_timesteps=100000)
        >>> action, log_prob, value = agent.select_action(state)
    """

    # Required attributes
    state_dim: int
    action_dim: int
    config: Dict[str, Any]
    device: torch.device

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action given state.

        Args:
            state: Current state observation
            deterministic: If True, use greedy action selection

        Returns:
            Tuple of (action, log_probability, state_value)
        """
        ...

    def train_on_env(
        self,
        env,
        total_timesteps: int,
        eval_env=None,
        eval_frequency: int = 10000,
        verbose: bool = True,
        mlflow_callback: Optional[Callable] = None,
        patience: Optional[int] = None,
        min_improvement: float = 0.01
    ) -> Dict[str, Any]:
        """
        Train agent on environment.

        Args:
            env: Training environment (Gymnasium-compatible)
            total_timesteps: Total timesteps to train
            eval_env: Optional evaluation environment
            eval_frequency: Evaluation frequency in timesteps
            verbose: Whether to log progress
            mlflow_callback: Optional callback for MLflow logging
            patience: Number of evaluations without improvement before early stopping
            min_improvement: Minimum improvement to reset patience counter

        Returns:
            Dictionary containing training metrics
        """
        ...

    def online_update(
        self,
        n_samples: int = 256,
        n_epochs: int = 5
    ) -> Optional[Dict[str, float]]:
        """
        Update agent from experience buffer (online learning).

        Samples from internal experience buffer and performs PPO update.

        Args:
            n_samples: Number of samples to use from buffer
            n_epochs: Number of training epochs

        Returns:
            Dictionary with loss metrics, or None if insufficient samples
        """
        ...

    def save(self, path: str) -> None:
        """
        Save agent checkpoint.

        Checkpoint should include:
            - model_state_dict: Network weights
            - optimizer_state_dict: Optimizer state
            - config: Agent configuration
            - training_history: Training metrics history
            - metadata: CheckpointMetadata with model_type, state_dim, action_dim

        Args:
            path: Path to save checkpoint
        """
        ...

    def load(self, path: str) -> None:
        """
        Load agent checkpoint.

        Note: Agent must be initialized with matching dimensions before loading.

        Args:
            path: Path to load checkpoint from
        """
        ...


# =============================================================================
# Type Aliases
# =============================================================================

# For type hints when accepting any model type
ModelType = PredictorModel | AgentModel
