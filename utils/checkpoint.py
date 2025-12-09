"""
Leap Trading System - Model Checkpoint Utilities
Standardized checkpoint structure for consistent model save/load across models.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch

logger = logging.getLogger(__name__)


# Standard checkpoint keys for consistency across models
CHECKPOINT_KEYS = {
    'MODEL_STATE': 'model_state_dict',
    'OPTIMIZER_STATE': 'optimizer_state_dict',
    'CONFIG': 'config',
    'TRAINING_HISTORY': 'training_history',
    'METADATA': 'metadata',
}

# Legacy key mappings for backward compatibility
LEGACY_KEY_MAPPINGS = {
    # TransformerPredictor legacy keys
    'train_losses': 'training_history',
    'val_losses': 'training_history',
    'input_dim': 'metadata',
    # PPOAgent legacy keys
    'network_state_dict': 'model_state_dict',
    'training_stats': 'training_history',
}


@dataclass
class CheckpointMetadata:
    """
    Metadata stored with model checkpoints.

    Captures architecture parameters and other model-specific info
    needed to reconstruct the model before loading weights.
    """
    model_type: str = ""  # 'transformer' or 'ppo'
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    version: str = "1.0"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'version': self.version,
            'extra': self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        return cls(
            model_type=data.get('model_type', ''),
            input_dim=data.get('input_dim'),
            output_dim=data.get('output_dim'),
            state_dim=data.get('state_dim'),
            action_dim=data.get('action_dim'),
            version=data.get('version', '1.0'),
            extra=data.get('extra', {}),
        )


@dataclass
class TrainingHistory:
    """
    Training history stored with model checkpoints.

    Unified structure for both transformer and PPO training metrics.
    """
    # Transformer-style losses
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)

    # PPO-style statistics
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropy_losses: List[float] = field(default_factory=list)
    total_losses: List[float] = field(default_factory=list)
    approx_kl: List[float] = field(default_factory=list)
    clip_fraction: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'total_losses': self.total_losses,
            'approx_kl': self.approx_kl,
            'clip_fraction': self.clip_fraction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingHistory':
        """Create from dictionary."""
        return cls(
            train_losses=data.get('train_losses', []),
            val_losses=data.get('val_losses', []),
            policy_losses=data.get('policy_losses', []),
            value_losses=data.get('value_losses', []),
            entropy_losses=data.get('entropy_losses', []),
            total_losses=data.get('total_losses', []),
            approx_kl=data.get('approx_kl', []),
            clip_fraction=data.get('clip_fraction', []),
        )

    @classmethod
    def from_legacy_transformer(
        cls,
        train_losses: List[float],
        val_losses: List[float]
    ) -> 'TrainingHistory':
        """Create from legacy TransformerPredictor format."""
        return cls(train_losses=train_losses, val_losses=val_losses)

    @classmethod
    def from_legacy_ppo(cls, training_stats: Dict[str, List[float]]) -> 'TrainingHistory':
        """Create from legacy PPOAgent format."""
        return cls(
            policy_losses=training_stats.get('policy_losses', []),
            value_losses=training_stats.get('value_losses', []),
            entropy_losses=training_stats.get('entropy_losses', []),
            total_losses=training_stats.get('total_losses', []),
            approx_kl=training_stats.get('approx_kl', []),
            clip_fraction=training_stats.get('clip_fraction', []),
        )


def create_checkpoint(
    model_state_dict: Dict[str, Any],
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    training_history: Optional[TrainingHistory] = None,
    metadata: Optional[CheckpointMetadata] = None
) -> Dict[str, Any]:
    """
    Create a standardized checkpoint dictionary.

    Args:
        model_state_dict: PyTorch model state dictionary
        optimizer_state_dict: Optional optimizer state dictionary
        config: Model configuration dictionary
        training_history: Training history object
        metadata: Checkpoint metadata object

    Returns:
        Standardized checkpoint dictionary
    """
    checkpoint = {
        CHECKPOINT_KEYS['MODEL_STATE']: model_state_dict,
    }

    if optimizer_state_dict is not None:
        checkpoint[CHECKPOINT_KEYS['OPTIMIZER_STATE']] = optimizer_state_dict

    if config is not None:
        checkpoint[CHECKPOINT_KEYS['CONFIG']] = config

    if training_history is not None:
        checkpoint[CHECKPOINT_KEYS['TRAINING_HISTORY']] = training_history.to_dict()

    if metadata is not None:
        checkpoint[CHECKPOINT_KEYS['METADATA']] = metadata.to_dict()

    return checkpoint


def save_checkpoint(
    path: str,
    model_state_dict: Dict[str, Any],
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    training_history: Optional[TrainingHistory] = None,
    metadata: Optional[CheckpointMetadata] = None
) -> None:
    """
    Save a standardized checkpoint to disk.

    Args:
        path: File path to save checkpoint
        model_state_dict: PyTorch model state dictionary
        optimizer_state_dict: Optional optimizer state dictionary
        config: Model configuration dictionary
        training_history: Training history object
        metadata: Checkpoint metadata object
    """
    checkpoint = create_checkpoint(
        model_state_dict=model_state_dict,
        optimizer_state_dict=optimizer_state_dict,
        config=config,
        training_history=training_history,
        metadata=metadata
    )
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Load a checkpoint with backward compatibility for legacy formats.

    Handles both new standardized format and legacy TransformerPredictor/PPOAgent formats.

    Args:
        path: File path to load checkpoint from
        device: Device to map tensors to

    Returns:
        Normalized checkpoint dictionary with standardized keys
    """
    # weights_only=False required for loading optimizer state and custom objects
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Normalize to standard format
    normalized = {}

    # Handle model state dict (check for legacy 'network_state_dict' key)
    if CHECKPOINT_KEYS['MODEL_STATE'] in checkpoint:
        normalized[CHECKPOINT_KEYS['MODEL_STATE']] = checkpoint[CHECKPOINT_KEYS['MODEL_STATE']]
    elif 'network_state_dict' in checkpoint:
        normalized[CHECKPOINT_KEYS['MODEL_STATE']] = checkpoint['network_state_dict']
        logger.debug("Loaded legacy 'network_state_dict' as 'model_state_dict'")

    # Handle optimizer state dict
    if CHECKPOINT_KEYS['OPTIMIZER_STATE'] in checkpoint:
        normalized[CHECKPOINT_KEYS['OPTIMIZER_STATE']] = checkpoint[CHECKPOINT_KEYS['OPTIMIZER_STATE']]

    # Handle config
    if CHECKPOINT_KEYS['CONFIG'] in checkpoint:
        normalized[CHECKPOINT_KEYS['CONFIG']] = checkpoint[CHECKPOINT_KEYS['CONFIG']]
    elif 'config' in checkpoint:
        normalized[CHECKPOINT_KEYS['CONFIG']] = checkpoint['config']

    # Handle training history (normalize from legacy formats)
    if CHECKPOINT_KEYS['TRAINING_HISTORY'] in checkpoint:
        history_data = checkpoint[CHECKPOINT_KEYS['TRAINING_HISTORY']]
        normalized[CHECKPOINT_KEYS['TRAINING_HISTORY']] = TrainingHistory.from_dict(history_data)
    elif 'training_stats' in checkpoint:
        # Legacy PPOAgent format
        normalized[CHECKPOINT_KEYS['TRAINING_HISTORY']] = TrainingHistory.from_legacy_ppo(
            checkpoint['training_stats']
        )
        logger.debug("Loaded legacy 'training_stats' as training_history")
    elif 'train_losses' in checkpoint or 'val_losses' in checkpoint:
        # Legacy TransformerPredictor format
        normalized[CHECKPOINT_KEYS['TRAINING_HISTORY']] = TrainingHistory.from_legacy_transformer(
            train_losses=checkpoint.get('train_losses', []),
            val_losses=checkpoint.get('val_losses', [])
        )
        logger.debug("Loaded legacy train_losses/val_losses as training_history")
    else:
        normalized[CHECKPOINT_KEYS['TRAINING_HISTORY']] = TrainingHistory()

    # Handle metadata
    if CHECKPOINT_KEYS['METADATA'] in checkpoint:
        normalized[CHECKPOINT_KEYS['METADATA']] = CheckpointMetadata.from_dict(
            checkpoint[CHECKPOINT_KEYS['METADATA']]
        )
    else:
        # Build metadata from legacy fields
        metadata = CheckpointMetadata()
        if 'input_dim' in checkpoint:
            metadata.input_dim = checkpoint['input_dim']
        normalized[CHECKPOINT_KEYS['METADATA']] = metadata

    logger.info(f"Checkpoint loaded from {path}")
    return normalized
