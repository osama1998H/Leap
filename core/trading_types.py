"""
Leap Trading System - Shared Trading Types
Common types and configurations used across trading environments.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional


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
