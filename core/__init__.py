"""Core module for Leap Trading System."""

from .data_pipeline import DataPipeline, FeatureEngineer, MarketData
from .feature_registry import FeatureRegistry, FeatureCategory, FeatureSpec
from .trading_env import TradingEnvironment, MultiSymbolTradingEnv
from .risk_manager import RiskManager, DynamicRiskManager, RiskLimits, PositionSizing
from .trading_types import (
    TradingError,
    InsufficientFundsError,
    OrderRejectedError,
    PositionError,
    BrokerConnectionError,
    DataPipelineError,
    RiskLimitExceededError,
)

__all__ = [
    'DataPipeline',
    'FeatureEngineer',
    'MarketData',
    # Feature Registry (ADR-0012)
    'FeatureRegistry',
    'FeatureCategory',
    'FeatureSpec',
    'TradingEnvironment',
    'MultiSymbolTradingEnv',
    'RiskManager',
    'DynamicRiskManager',
    'RiskLimits',
    'PositionSizing',
    # Exception hierarchy
    'TradingError',
    'InsufficientFundsError',
    'OrderRejectedError',
    'PositionError',
    'BrokerConnectionError',
    'DataPipelineError',
    'RiskLimitExceededError',
]
