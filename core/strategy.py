"""
Leap Trading System - Trading Strategy Pattern

Defines abstract base class for trading strategies and provides
default implementations for consistent signal generation across
backtesting and live trading.

See ADR-0011 for design rationale.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
import numpy as np
import pandas as pd

from core.trading_types import Trade, Action
from core.order_manager import SignalType, TradingSignal

logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Signal Output
# =============================================================================

@dataclass
class StrategySignal:
    """
    Unified signal output from strategies.

    Used by both backtesting and live trading for consistency.
    Provides conversion methods for backward compatibility.
    """
    action: SignalType
    symbol: str = ""
    confidence: float = 1.0
    stop_loss_pips: Optional[float] = None
    take_profit_pips: Optional[float] = None
    risk_percent: Optional[float] = None
    predicted_return: float = 0.0
    agent_action: Optional[Action] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_backtest_dict(self) -> Dict[str, Any]:
        """
        Convert to backtester-compatible dictionary format.

        For backward compatibility with Backtester.run() callable interface.

        Returns:
            Dictionary with 'action', 'stop_loss_pips', 'take_profit_pips'
        """
        action_str = self.action.value if self.action != SignalType.HOLD else 'hold'

        return {
            'action': action_str,
            'stop_loss_pips': self.stop_loss_pips or 50.0,
            'take_profit_pips': self.take_profit_pips or 100.0
        }

    def to_trading_signal(self) -> TradingSignal:
        """
        Convert to TradingSignal for order execution.

        Returns:
            TradingSignal instance for OrderManager
        """
        return TradingSignal(
            signal_type=self.action,
            symbol=self.symbol,
            confidence=self.confidence,
            stop_loss_pips=self.stop_loss_pips,
            take_profit_pips=self.take_profit_pips,
            risk_percent=self.risk_percent,
            source="strategy",
            metadata={
                'predicted_return': self.predicted_return,
                'agent_action': self.agent_action.name if self.agent_action else None,
                **self.metadata
            }
        )

    @property
    def is_entry(self) -> bool:
        """Check if this is an entry signal."""
        return self.action in [SignalType.BUY, SignalType.SELL]

    @property
    def is_exit(self) -> bool:
        """Check if this is an exit signal."""
        return self.action in [SignalType.CLOSE, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]

    @property
    def is_hold(self) -> bool:
        """Check if this is a hold signal."""
        return self.action == SignalType.HOLD


# =============================================================================
# Strategy Configuration
# =============================================================================

@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    # Confidence thresholds
    min_confidence: float = 0.6

    # Prediction thresholds
    prediction_threshold: float = 0.001

    # Default SL/TP
    default_sl_pips: float = 50.0
    default_tp_pips: float = 100.0

    # Risk management
    risk_per_trade: float = 0.02

    # Model parameters
    lookback_window: int = 60


# =============================================================================
# Abstract Base Strategy
# =============================================================================

class TradingStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Defines the contract for all strategy implementations.
    Used by both Backtester and AutoTrader for consistent signal generation.

    Implementations must override:
    - name: Strategy identifier
    - generate_signal(): Main signal generation method

    Optional overrides:
    - on_trade_opened(): Callback when trade opens
    - on_trade_closed(): Callback when trade closes
    - reset(): Reset strategy state

    Example:
        >>> class MyStrategy(TradingStrategy):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_custom_strategy"
        ...
        ...     def generate_signal(self, market_data, positions, **kwargs):
        ...         # Custom signal logic
        ...         return StrategySignal(action=SignalType.BUY, symbol="EURUSD")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Strategy identifier.

        Returns:
            Unique name for this strategy
        """
        pass

    @abstractmethod
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        positions: List[Trade],
        **kwargs
    ) -> StrategySignal:
        """
        Generate a trading signal based on market data.

        This is the main method that defines strategy behavior.

        Args:
            market_data: OHLCV + features DataFrame (most recent data)
            positions: Current open positions
            **kwargs: Additional strategy-specific parameters
                - predictor: TransformerPredictor instance (optional)
                - agent: PPOAgent instance (optional)
                - symbol: Trading symbol (optional)
                - open_status: Whether new positions can be opened (optional)

        Returns:
            StrategySignal with action and parameters
        """
        pass

    def on_trade_opened(self, trade: Trade) -> None:
        """
        Callback when a trade is opened.

        Override to implement custom logic on trade entry.

        Args:
            trade: The opened trade
        """
        pass

    def on_trade_closed(self, trade: Trade) -> None:
        """
        Callback when a trade is closed.

        Override to implement custom logic on trade exit.

        Args:
            trade: The closed trade
        """
        pass

    def reset(self) -> None:
        """
        Reset strategy state.

        Override to clear any internal state between backtests
        or trading sessions.
        """
        pass


# =============================================================================
# Combined Predictor + Agent Strategy
# =============================================================================

class CombinedPredictorAgentStrategy(TradingStrategy):
    """
    Default strategy combining Transformer predictor with PPO agent.

    This strategy:
    1. Gets Transformer prediction for price direction
    2. Gets PPO agent action based on observation
    3. Combines both signals with validation

    The combination logic:
    - Agent decision takes priority
    - Prediction validates/contradicts the decision
    - Strong contradiction results in HOLD
    - Weak prediction trusts the agent

    This consolidates the logic previously in:
    - AutoTrader._combine_signals()
    - cli/system.py backtest strategy function

    Example:
        >>> strategy = CombinedPredictorAgentStrategy(
        ...     predictor=predictor,
        ...     agent=agent,
        ...     config=StrategyConfig(min_confidence=0.7)
        ... )
        >>> signal = strategy.generate_signal(data, positions)
    """

    def __init__(
        self,
        predictor=None,
        agent=None,
        config: Optional[StrategyConfig] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize combined strategy.

        Args:
            predictor: TransformerPredictor instance (optional)
            agent: PPOAgent instance (optional)
            config: Strategy configuration
            feature_names: Feature names for observation building
        """
        self.predictor = predictor
        self.agent = agent
        self.config = config or StrategyConfig()
        self.feature_names = feature_names or []

        # Internal state
        self._last_prediction: Optional[Dict] = None
        self._last_action: Optional[Action] = None

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "combined_predictor_agent"

    def _get_feature_columns(self, market_data: pd.DataFrame) -> List[str]:
        """
        Get feature columns for inference, matching training data format.

        Training data (prepare_sequences) uses:
            np.column_stack([open, high, low, close, volume, features])

        This method ensures inference uses the SAME feature order:
            [open, high, low, close, volume, ...computed_features]

        IMPORTANT: OHLCV columns are ALWAYS included first, regardless of
        what feature_names contains. The model was trained with OHLCV data
        and inference must match.

        Args:
            market_data: DataFrame with OHLCV and computed features

        Returns:
            List of column names in correct order for model input
        """
        # OHLCV columns must ALWAYS be included first to match training format
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        metadata_cols = ['time', 'datetime']

        # Get OHLCV columns that exist in the data
        available_ohlcv = [col for col in ohlcv_cols if col in market_data.columns]

        # Determine computed feature columns
        if self.feature_names:
            # Use provided feature names for computed features
            # Filter to only include names that exist in the data
            computed_cols = [col for col in self.feature_names if col in market_data.columns]
        else:
            # Auto-detect computed features (everything except OHLCV and metadata)
            computed_cols = [
                col for col in market_data.columns
                if col not in ohlcv_cols + metadata_cols
            ]

        # Return in correct order: OHLCV + computed features
        return available_ohlcv + computed_cols

    def generate_signal(
        self,
        market_data: pd.DataFrame,
        positions: List[Trade],
        **kwargs
    ) -> StrategySignal:
        """
        Generate signal combining Transformer and PPO outputs.

        Args:
            market_data: OHLCV + features DataFrame
            positions: Current open positions
            **kwargs:
                - predictor: Override predictor
                - agent: Override agent
                - symbol: Trading symbol
                - open_status: Whether new positions can be opened (default: True)

        Returns:
            StrategySignal with combined action
        """
        # Allow override of predictor/agent from kwargs
        predictor = kwargs.get('predictor', self.predictor)
        agent = kwargs.get('agent', self.agent)
        symbol = kwargs.get('symbol', '')
        open_status = kwargs.get('open_status', True)

        # Default values
        predicted_return = 0.0
        prediction_confidence = 0.5
        agent_action = Action.HOLD

        # Get Transformer prediction
        if predictor is not None and len(market_data) >= self.config.lookback_window:
            try:
                prediction = self._get_prediction(predictor, market_data)
                if prediction is not None:
                    predicted_return = prediction.get('prediction', np.array([[0.0]]))[0, 0]
                    uncertainty = prediction.get('uncertainty', 0.5)
                    prediction_confidence = max(0.0, min(1.0, 1.0 - uncertainty))
                    self._last_prediction = prediction
            except Exception as e:
                logger.warning(f"Prediction error: {e}")

        # Get PPO agent action
        if agent is not None and len(market_data) >= self.config.lookback_window:
            try:
                # Get account state from kwargs (passed by backtester/auto_trader)
                account_state = kwargs.get('account_state', None)
                obs = self._build_observation(market_data, positions, account_state)
                if obs is not None:
                    action_idx, _, _ = agent.select_action(obs, deterministic=True)
                    agent_action = Action(action_idx)
                    self._last_action = agent_action
            except Exception as e:
                logger.warning(f"Agent action error: {e}")

        # Combine signals
        signal_type = self._combine_signals(
            predicted_return=predicted_return,
            agent_action=agent_action,
            confidence=prediction_confidence,
            open_status=open_status
        )

        return StrategySignal(
            action=signal_type,
            symbol=symbol,
            confidence=prediction_confidence,
            stop_loss_pips=self.config.default_sl_pips,
            take_profit_pips=self.config.default_tp_pips,
            risk_percent=self.config.risk_per_trade,
            predicted_return=predicted_return,
            agent_action=agent_action,
            metadata={
                'strategy': self.name,
                'uncertainty': 1.0 - prediction_confidence
            }
        )

    def _get_prediction(
        self,
        predictor,
        market_data: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Get prediction from Transformer model.

        Args:
            predictor: TransformerPredictor instance
            market_data: DataFrame with features

        Returns:
            Prediction dictionary or None
        """
        # Get feature columns - MUST include OHLCV to match training data format
        # Feature order: [open, high, low, close, volume, ...computed_features]
        # This matches prepare_sequences() in data_pipeline.py which uses:
        # np.column_stack([open, high, low, close, volume, features])
        feature_cols = self._get_feature_columns(market_data)

        # Extract features
        features = market_data[feature_cols].tail(self.config.lookback_window).values

        # Handle NaN values
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, nan=0.0)

        # Reshape for model: (batch=1, seq_len, features)
        X = features.reshape(1, self.config.lookback_window, -1)

        # Validate feature dimension matches model expectation (if input_dim is available)
        if hasattr(predictor, 'input_dim'):
            expected_dim = predictor.input_dim
            # Only validate if input_dim is a valid integer (not a Mock object)
            if isinstance(expected_dim, int) and X.shape[2] != expected_dim:
                raise ValueError(
                    f"Feature dimension mismatch: model expects {expected_dim} features, "
                    f"but inference data has {X.shape[2]}. "
                    f"Ensure training and inference use the same feature set (OHLCV + computed)."
                )

        # Get prediction
        return predictor.predict(X, return_uncertainty=True)

    def _build_observation(
        self,
        market_data: pd.DataFrame,
        positions: List[Trade],
        account_state: Optional[Dict[str, float]] = None
    ) -> Optional[np.ndarray]:
        """
        Build observation for PPO agent.

        The observation must match the training environment format:
        - Market features: (window_size × n_features) flattened
        - Account features: 8 values (balance_norm, equity_norm, n_positions,
          has_long, has_short, unrealized_pnl_norm, max_drawdown, total_pnl_norm)

        Args:
            market_data: DataFrame with features
            positions: Current positions
            account_state: Account state dict with keys:
                - balance: Current balance
                - equity: Current equity
                - initial_balance: Initial balance for normalization
                - unrealized_pnl: Current unrealized PnL
                - max_drawdown: Maximum drawdown ratio
                - total_pnl: Total realized PnL

        Returns:
            Observation array or None
        """
        # Get feature columns - MUST include OHLCV to match training data format
        # Feature order: [open, high, low, close, volume, ...computed_features]
        feature_cols = self._get_feature_columns(market_data)

        # Extract latest features
        features = market_data[feature_cols].tail(self.config.lookback_window).values

        # Flatten for agent
        market_obs = features.flatten()

        # Handle NaN values
        if np.any(np.isnan(market_obs)):
            market_obs = np.nan_to_num(market_obs, nan=0.0)

        # Build account observation (8 features to match training environment)
        # See core/trading_env_base.py:_get_account_observation()
        account_obs = self._build_account_observation(positions, account_state)

        # Concatenate market and account observations
        obs = np.concatenate([market_obs, account_obs])

        return obs.astype(np.float32)

    def _build_account_observation(
        self,
        positions: List[Trade],
        account_state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Build account observation matching TradingEnvironment format.

        The 8 account features are:
        1. balance_norm: (balance / initial_balance - 1.0), normalized
        2. equity_norm: (equity / initial_balance - 1.0), normalized
        3. n_positions: Number of open positions
        4. has_long: 1.0 if has long position, else 0.0
        5. has_short: 1.0 if has short position, else 0.0
        6. unrealized_pnl_norm: unrealized_pnl / initial_balance, normalized
        7. max_drawdown: Maximum drawdown ratio
        8. total_pnl_norm: total_pnl / initial_balance, normalized

        Args:
            positions: Current open positions
            account_state: Account state dict (if None, uses defaults)

        Returns:
            8-element account observation array
        """
        # Default account state if not provided
        if account_state is None:
            account_state = {
                'balance': 10000.0,
                'equity': 10000.0,
                'initial_balance': 10000.0,
                'unrealized_pnl': 0.0,
                'max_drawdown': 0.0,
                'total_pnl': 0.0
            }

        initial_balance = account_state.get('initial_balance', 10000.0)
        balance = account_state.get('balance', initial_balance)
        equity = account_state.get('equity', balance)
        unrealized_pnl = account_state.get('unrealized_pnl', 0.0)
        max_drawdown = account_state.get('max_drawdown', 0.0)
        total_pnl = account_state.get('total_pnl', 0.0)

        # Check position directions
        has_long = any(p.direction == 'long' for p in positions) if positions else False
        has_short = any(p.direction == 'short' for p in positions) if positions else False

        # Normalize using log-scale for unbounded values (matches trading_env_base.py)
        def normalize_ratio(x: float) -> float:
            """Log-scale normalization for unbounded values."""
            if x >= 0:
                return np.log1p(x)
            else:
                return -np.log1p(-x)

        return np.array([
            normalize_ratio(balance / initial_balance - 1.0),
            normalize_ratio(equity / initial_balance - 1.0),
            float(len(positions)) if positions else 0.0,
            1.0 if has_long else 0.0,
            1.0 if has_short else 0.0,
            normalize_ratio(unrealized_pnl / initial_balance),
            max_drawdown,
            normalize_ratio(total_pnl / initial_balance)
        ], dtype=np.float32)

    def _combine_signals(
        self,
        predicted_return: float,
        agent_action: Action,
        confidence: float,
        open_status: bool = True
    ) -> SignalType:
        """
        Combine prediction and agent action into final signal.

        This is the core signal combination logic, consolidated from
        AutoTrader._combine_signals() and cli/system.py strategy.

        Args:
            predicted_return: Predicted price return from Transformer
            agent_action: Action from PPO agent
            confidence: Prediction confidence (1 - uncertainty)
            open_status: Whether new positions can be opened

        Returns:
            Final SignalType
        """
        # Check confidence threshold
        if confidence < self.config.min_confidence:
            return SignalType.HOLD

        # Check if new positions allowed
        if not open_status:
            if agent_action == Action.CLOSE:
                return SignalType.CLOSE
            return SignalType.HOLD

        # Close action always passes through
        if agent_action == Action.CLOSE:
            return SignalType.CLOSE

        # BUY signal logic
        if agent_action == Action.BUY:
            if predicted_return >= self.config.prediction_threshold:
                # Agreement - strong buy signal
                return SignalType.BUY
            elif predicted_return < -self.config.prediction_threshold:
                # Strong contradiction - hold
                return SignalType.HOLD
            else:
                # Weak prediction - trust agent
                return SignalType.BUY

        # SELL signal logic
        if agent_action == Action.SELL:
            if predicted_return <= -self.config.prediction_threshold:
                # Agreement - strong sell signal
                return SignalType.SELL
            elif predicted_return > self.config.prediction_threshold:
                # Strong contradiction - hold
                return SignalType.HOLD
            else:
                # Weak prediction - trust agent
                return SignalType.SELL

        # Default: HOLD
        return SignalType.HOLD

    def reset(self) -> None:
        """Reset strategy state."""
        self._last_prediction = None
        self._last_action = None


# =============================================================================
# Strategy Adapter for Callable Functions
# =============================================================================

class CallableStrategyAdapter(TradingStrategy):
    """
    Adapter to wrap callable functions as TradingStrategy.

    Provides backward compatibility for legacy callable strategies
    used with Backtester.run().

    Example:
        >>> def my_strategy(data, predictor, agent, positions):
        ...     return {'action': 'buy', 'stop_loss_pips': 50, 'take_profit_pips': 100}
        ...
        >>> strategy = CallableStrategyAdapter(my_strategy, "my_strategy")
    """

    def __init__(
        self,
        callable_fn: Callable,
        strategy_name: str = "callable_strategy"
    ):
        """
        Initialize adapter.

        Args:
            callable_fn: Strategy function with signature:
                (data, predictor, agent, positions) -> dict
            strategy_name: Name for this strategy
        """
        self._callable = callable_fn
        self._name = strategy_name

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return self._name

    def generate_signal(
        self,
        market_data: pd.DataFrame,
        positions: List[Trade],
        **kwargs
    ) -> StrategySignal:
        """
        Generate signal by calling wrapped function.

        Args:
            market_data: OHLCV + features DataFrame
            positions: Current open positions
            **kwargs: Passed to callable

        Returns:
            StrategySignal converted from callable result
        """
        predictor = kwargs.get('predictor')
        agent = kwargs.get('agent')
        symbol = kwargs.get('symbol', '')

        # Call the wrapped function
        result = self._callable(
            market_data,
            predictor=predictor,
            agent=agent,
            positions=positions
        )

        # Convert result to StrategySignal
        action_str = result.get('action', 'hold').lower()

        # Map string action to SignalType
        action_map = {
            'buy': SignalType.BUY,
            'sell': SignalType.SELL,
            'close': SignalType.CLOSE,
            'close_long': SignalType.CLOSE_LONG,
            'close_short': SignalType.CLOSE_SHORT,
            'hold': SignalType.HOLD,
        }
        action = action_map.get(action_str, SignalType.HOLD)

        return StrategySignal(
            action=action,
            symbol=symbol,
            stop_loss_pips=result.get('stop_loss_pips', 50.0),
            take_profit_pips=result.get('take_profit_pips', 100.0),
            metadata={'source': 'callable_adapter'}
        )


# =============================================================================
# Utility Functions
# =============================================================================

def create_strategy(
    strategy_type: str = 'combined',
    predictor=None,
    agent=None,
    config: Optional[StrategyConfig] = None,
    **kwargs
) -> TradingStrategy:
    """
    Factory function to create strategy instances.

    Args:
        strategy_type: Type of strategy ('combined', 'callable')
        predictor: TransformerPredictor instance
        agent: PPOAgent instance
        config: Strategy configuration
        **kwargs: Additional arguments

    Returns:
        TradingStrategy instance

    Example:
        >>> strategy = create_strategy('combined', predictor=pred, agent=ag)
    """
    if strategy_type.lower() == 'combined':
        return CombinedPredictorAgentStrategy(
            predictor=predictor,
            agent=agent,
            config=config,
            feature_names=kwargs.get('feature_names')
        )
    elif strategy_type.lower() == 'callable':
        callable_fn = kwargs.get('callable_fn')
        if callable_fn is None:
            raise ValueError("callable_fn required for callable strategy type")
        return CallableStrategyAdapter(
            callable_fn=callable_fn,
            strategy_name=kwargs.get('strategy_name', 'callable_strategy')
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


__all__ = [
    # Base class
    'TradingStrategy',

    # Implementations
    'CombinedPredictorAgentStrategy',
    'CallableStrategyAdapter',

    # Data classes
    'StrategySignal',
    'StrategyConfig',

    # Factory
    'create_strategy',
]
