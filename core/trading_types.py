"""
Leap Trading System - Shared Trading Types
Common types and configurations used across trading environments.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from typing import List, Optional, Union


class Action(IntEnum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


@dataclass
class EnvConfig:
    """Shared configuration for trading environments."""
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
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0  # Must be set externally; not updated by update_from_trade()

    @property
    def win_rate(self) -> float:
        """Calculate win rate as a percentage."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
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
