"""Training module for Leap Trading System."""

from .online_learning import OnlineLearningManager, AdaptiveTrainer
from .trainer import ModelTrainer
from .online_interface import (
    OnlineLearningData,
    OnlineLearningResult,
    OnlineLearningAdapter,
    create_online_learner
)

__all__ = [
    'OnlineLearningManager',
    'AdaptiveTrainer',
    'ModelTrainer',
    # Unified online learning interface (MAJOR-7)
    'OnlineLearningData',
    'OnlineLearningResult',
    'OnlineLearningAdapter',
    'create_online_learner',
]
