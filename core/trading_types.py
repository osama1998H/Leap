"""
Leap Trading System - Shared Trading Types
Common types and configurations used across trading environments.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import List, Optional, Union


class Action(IntEnum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


# =============================================================================
# Exception Hierarchy for Trading Operations
# =============================================================================
# Use these custom exceptions for better error handling across the codebase.
# All trading-related exceptions inherit from TradingError for unified catching.
# Always log exceptions using logger.exception() when catching.


class TradingError(Exception):
    """
    Base exception for all trading-related errors.

    All custom trading exceptions inherit from this class, allowing
    unified exception handling:

        try:
            execute_trade(...)
        except TradingError as e:
            logger.exception("Trading operation failed")
            handle_error(e)
    """
    pass


class InsufficientFundsError(TradingError):
    """
    Raised when account balance is insufficient for a trade.

    Example:
        if balance < required_margin:
            raise InsufficientFundsError(
                f"Balance {balance} insufficient for margin {required_margin}"
            )
    """
    pass


class OrderRejectedError(TradingError):
    """
    Raised when a trade order is rejected by broker or risk manager.

    Example:
        if not risk_manager.approve_order(order):
            raise OrderRejectedError(f"Order rejected: {order}")
    """
    pass


class PositionError(TradingError):
    """
    Raised for position-related errors (not found, already closed, etc.).

    Example:
        if position.is_closed:
            raise PositionError(f"Position {position.id} is already closed")
    """
    pass


class BrokerConnectionError(TradingError):
    """
    Raised when broker connection fails or is lost.

    Example:
        if not mt5.initialize():
            raise BrokerConnectionError("Failed to connect to MT5")
    """
    pass


class DataPipelineError(TradingError):
    """
    Raised for data fetching or processing errors.

    Example:
        if data is None or len(data) == 0:
            raise DataPipelineError(f"No data available for {symbol}")
    """
    pass


class RiskLimitExceededError(TradingError):
    """
    Raised when a trade would exceed risk limits.

    Example:
        if drawdown > max_drawdown:
            raise RiskLimitExceededError(
                f"Max drawdown {max_drawdown} exceeded: {drawdown}"
            )
    """
    pass


@dataclass
class EnvConfig:
    """
    Shared configuration for trading environments.

    Use EnvConfig directly or create via from_params() factory method.
    This is the single source of truth for environment configuration defaults.
    """
    initial_balance: float = 10000.0
    commission: float = 0.0001  # 0.01% per trade
    spread: float = 0.0002  # 2 pips for forex
    slippage: float = 0.0001  # 1 pip slippage
    leverage: int = 100
    max_position_size: float = 0.1  # Max 10% of balance
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    window_size: int = 60
    max_drawdown_threshold: float = 0.5  # 50% max drawdown terminates episode
    max_episode_steps: int = 2000  # Max steps per episode (prevents extremely long episodes)

    # === Reward Shaping Parameters (v2 - Log-Return Based) ===
    # New reward formula: r = log(E_t/E_{t-1}) - c_tc*|Δw| - λ_dd*max(0, DD-DD_max) - λ_vol*σ²

    # Transaction cost penalty: penalizes position changes to discourage overtrading
    # |Δw| = absolute change in position weight (exposure ratio)
    tc_penalty_scale: float = 1.0  # Multiplier for transaction cost penalty

    # Threshold-based drawdown penalty: only penalizes when DD exceeds threshold
    dd_threshold: float = 0.05  # 5% drawdown tolerance before penalty kicks in
    dd_penalty_scale: float = 10.0  # Penalty strength for excess drawdown

    # Optional volatility penalty: discourages high-variance strategies
    vol_penalty_scale: float = 0.0  # Disabled by default (set > 0 to enable)
    vol_window: int = 20  # Rolling window for return volatility estimation

    # Reward clipping for numerical stability (wider range for log-returns)
    reward_clip: float = 10.0  # Clip reward to [-reward_clip, +reward_clip]

    # Reward normalization: use running mean/std normalization in PPO
    normalize_reward: bool = True  # Enable running reward normalization

    # === Legacy Parameters (kept for backward compatibility) ===
    return_scale: float = 1.0  # Not used in v2, log-returns are scale-stable
    drawdown_penalty_scale: float = 20.0  # Legacy: use dd_penalty_scale instead
    recovery_bonus_scale: float = 20.0  # Legacy: not used in v2
    holding_cost: float = 0.0  # Legacy: use tc_penalty_scale instead

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # === New v2 reward parameter validation ===
        if self.tc_penalty_scale < 0:
            raise ValueError(f"tc_penalty_scale must be >= 0, got {self.tc_penalty_scale}")
        if self.dd_threshold < 0 or self.dd_threshold > 1:
            raise ValueError(f"dd_threshold must be in [0, 1], got {self.dd_threshold}")
        if self.dd_penalty_scale < 0:
            raise ValueError(f"dd_penalty_scale must be >= 0, got {self.dd_penalty_scale}")
        if self.vol_penalty_scale < 0:
            raise ValueError(f"vol_penalty_scale must be >= 0, got {self.vol_penalty_scale}")
        if self.vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {self.vol_window}")
        if self.reward_clip <= 0:
            raise ValueError(f"reward_clip must be > 0, got {self.reward_clip}")

        # === Legacy parameter validation (backward compatibility) ===
        if self.holding_cost < 0:
            raise ValueError(
                f"holding_cost must be >= 0 (it's a cost magnitude), got {self.holding_cost}. "
                f"The cost is applied as -holding_cost * n_positions internally."
            )
        if self.return_scale < 0:
            raise ValueError(f"return_scale must be >= 0, got {self.return_scale}")
        if self.drawdown_penalty_scale < 0:
            raise ValueError(f"drawdown_penalty_scale must be >= 0, got {self.drawdown_penalty_scale}")
        if self.recovery_bonus_scale < 0:
            raise ValueError(f"recovery_bonus_scale must be >= 0, got {self.recovery_bonus_scale}")

    @classmethod
    def from_params(
        cls,
        initial_balance: Optional[float] = None,
        commission: Optional[float] = None,
        spread: Optional[float] = None,
        slippage: Optional[float] = None,
        leverage: Optional[int] = None,
        max_position_size: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        window_size: Optional[int] = None,
        max_drawdown_threshold: Optional[float] = None,
        max_episode_steps: Optional[int] = None,
        # New v2 reward shaping parameters
        tc_penalty_scale: Optional[float] = None,
        dd_threshold: Optional[float] = None,
        dd_penalty_scale: Optional[float] = None,
        vol_penalty_scale: Optional[float] = None,
        vol_window: Optional[int] = None,
        reward_clip: Optional[float] = None,
        normalize_reward: Optional[bool] = None,
        # Legacy reward parameters (backward compatibility)
        return_scale: Optional[float] = None,
        drawdown_penalty_scale: Optional[float] = None,
        recovery_bonus_scale: Optional[float] = None,
        holding_cost: Optional[float] = None
    ) -> 'EnvConfig':
        """
        Factory method to create EnvConfig from individual parameters.

        Only non-None parameters override the defaults. This provides a clean
        interface for creating configs from keyword arguments.

        Example:
            config = EnvConfig.from_params(initial_balance=50000.0, leverage=50)
        """
        defaults = cls()
        return cls(
            initial_balance=initial_balance if initial_balance is not None else defaults.initial_balance,
            commission=commission if commission is not None else defaults.commission,
            spread=spread if spread is not None else defaults.spread,
            slippage=slippage if slippage is not None else defaults.slippage,
            leverage=leverage if leverage is not None else defaults.leverage,
            max_position_size=max_position_size if max_position_size is not None else defaults.max_position_size,
            stop_loss_pct=stop_loss_pct if stop_loss_pct is not None else defaults.stop_loss_pct,
            take_profit_pct=take_profit_pct if take_profit_pct is not None else defaults.take_profit_pct,
            window_size=window_size if window_size is not None else defaults.window_size,
            max_drawdown_threshold=max_drawdown_threshold if max_drawdown_threshold is not None else defaults.max_drawdown_threshold,
            max_episode_steps=max_episode_steps if max_episode_steps is not None else defaults.max_episode_steps,
            # New v2 reward parameters
            tc_penalty_scale=tc_penalty_scale if tc_penalty_scale is not None else defaults.tc_penalty_scale,
            dd_threshold=dd_threshold if dd_threshold is not None else defaults.dd_threshold,
            dd_penalty_scale=dd_penalty_scale if dd_penalty_scale is not None else defaults.dd_penalty_scale,
            vol_penalty_scale=vol_penalty_scale if vol_penalty_scale is not None else defaults.vol_penalty_scale,
            vol_window=vol_window if vol_window is not None else defaults.vol_window,
            reward_clip=reward_clip if reward_clip is not None else defaults.reward_clip,
            normalize_reward=normalize_reward if normalize_reward is not None else defaults.normalize_reward,
            # Legacy parameters
            return_scale=return_scale if return_scale is not None else defaults.return_scale,
            drawdown_penalty_scale=drawdown_penalty_scale if drawdown_penalty_scale is not None else defaults.drawdown_penalty_scale,
            recovery_bonus_scale=recovery_bonus_scale if recovery_bonus_scale is not None else defaults.recovery_bonus_scale,
            holding_cost=holding_cost if holding_cost is not None else defaults.holding_cost
        )


@dataclass
class Position:
    """Represents an open trading position."""
    type: str  # 'long' or 'short'
    entry_price: float
    size: float
    entry_time: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None


@dataclass
class TradingState:
    """Current trading state."""
    balance: float
    equity: float
    positions: List[Position] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0  # Sum of all winning trade profits
    gross_loss: float = 0.0    # Sum of all losing trade losses (absolute value)
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0


@dataclass
class LiveTradingState(TradingState):
    """Extended trading state for live trading."""
    # Real account data
    real_balance: float = 0.0
    real_equity: float = 0.0
    real_margin: float = 0.0
    real_free_margin: float = 0.0
    real_margin_level: float = 0.0

    # Trading control
    open_status: bool = True  # Can we open new positions?
    close_only: bool = False  # Only allow closing positions

    # Session statistics
    session_start_balance: float = 0.0
    session_pnl: float = 0.0
    session_trades: int = 0


@dataclass
class Trade:
    """
    Represents a single trade (open or closed).

    Consolidated dataclass used across backtester, auto_trader, and other modules.
    """
    entry_time: Union[datetime, int]
    entry_price: float
    direction: str  # 'long' or 'short'
    size: float
    exit_time: Optional[Union[datetime, int]] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    status: str = 'open'  # 'open', 'closed', 'stopped'

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.status in ('closed', 'stopped')

    @property
    def is_winning(self) -> bool:
        """Check if trade is a winner (closed with positive PnL)."""
        return self.is_closed and self.pnl > 0


@dataclass
class TradeStatistics:
    """
    Reusable trade statistics.

    Used by TradingState, TradingSession, and BacktestResult to avoid duplication.

    Note: max_drawdown requires equity curve tracking and cannot be computed from
    individual trade PnL alone. When using update_from_trade(), max_drawdown will
    not be updated. Set max_drawdown explicitly when you have access to equity data,
    or use TradingSession which tracks drawdown via account balance.
    """
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0  # Trades with pnl == 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0  # Must be set externally; not updated by update_from_trade()

    @property
    def win_rate(self) -> float:
        """Calculate win rate as a ratio (0.0 to 1.0)."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Returns:
            float('inf') when there are only winning trades (gross_loss == 0).
            0.0 when there are no trades or only losing/breakeven trades.
        """
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss

    @property
    def avg_winner(self) -> float:
        """Calculate average winning trade."""
        if self.winning_trades == 0:
            return 0.0
        return self.gross_profit / self.winning_trades

    @property
    def avg_loser(self) -> float:
        """Calculate average losing trade."""
        if self.losing_trades == 0:
            return 0.0
        return self.gross_loss / self.losing_trades

    def update_from_trade(self, trade: Trade) -> None:
        """Update statistics from a closed trade."""
        if not trade.is_closed:
            return

        self.total_trades += 1
        self.total_pnl += trade.pnl

        if trade.pnl > 0:
            self.winning_trades += 1
            self.gross_profit += trade.pnl
        elif trade.pnl < 0:
            self.losing_trades += 1
            self.gross_loss += abs(trade.pnl)
        else:
            self.breakeven_trades += 1
