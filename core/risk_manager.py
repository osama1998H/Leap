"""
Leap Trading System - Risk Management Module
Implements comprehensive risk management for trading.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_size_pct: float = 0.02  # Max 2% of account per position
    max_total_exposure_pct: float = 0.10  # Max 10% total exposure
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss
    max_weekly_loss_pct: float = 0.10  # Max 10% weekly loss
    max_drawdown_pct: float = 0.15  # Max 15% drawdown
    max_consecutive_losses: int = 5  # Max consecutive losing trades
    max_open_positions: int = 5  # Max number of open positions
    max_correlation: float = 0.7  # Max correlation between positions
    min_risk_reward: float = 1.5  # Minimum risk/reward ratio


@dataclass
class PositionSizing:
    """Position sizing parameters."""
    method: str = 'kelly'  # 'fixed', 'percent', 'kelly', 'volatility'
    fixed_size: float = 0.01  # Fixed lot size
    percent_risk: float = 0.02  # Percent of account to risk
    kelly_fraction: float = 0.5  # Fraction of Kelly to use
    max_leverage: int = 10  # Maximum leverage


@dataclass
class RiskState:
    """Current risk state."""
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_losses: int = 0
    open_positions: int = 0
    total_exposure: float = 0.0
    is_trading_allowed: bool = True
    halt_reason: Optional[str] = None


class RiskManager:
    """
    Comprehensive risk management system.

    Features:
    - Position sizing (multiple methods)
    - Drawdown monitoring
    - Daily/weekly loss limits
    - Correlation-based exposure limits
    - Dynamic stop loss/take profit
    - Circuit breakers
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        limits: Optional[RiskLimits] = None,
        sizing: Optional[PositionSizing] = None
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.limits = limits or RiskLimits()
        self.sizing = sizing or PositionSizing()

        # State tracking
        self.state = RiskState()
        self.peak_equity = initial_balance

        # History tracking
        self.equity_history: deque = deque(maxlen=10000)
        self.trade_history: deque = deque(maxlen=1000)
        self.daily_pnl_history: deque = deque(maxlen=365)

        # Time tracking for daily/weekly resets
        self.last_reset_date = datetime.now().date()
        self.week_start_balance = initial_balance

        # Win rate tracking for Kelly
        self.wins = 0
        self.losses = 0
        self.avg_win = 0.0
        self.avg_loss = 0.0

    def update_balance(self, new_balance: float):
        """Update current balance and recalculate risk state."""
        pnl = new_balance - self.current_balance
        self.current_balance = new_balance

        # Update peak and drawdown
        if new_balance > self.peak_equity:
            self.peak_equity = new_balance

        self.state.current_drawdown = (self.peak_equity - new_balance) / self.peak_equity

        # Update daily PnL
        self.state.daily_pnl += pnl

        # Check for date rollover
        today = datetime.now().date()
        if today != self.last_reset_date:
            self._handle_date_rollover(today)

        # Update equity history
        self.equity_history.append({
            'timestamp': datetime.now(),
            'balance': new_balance,
            'drawdown': self.state.current_drawdown
        })

        # Check risk limits
        self._check_risk_limits()

    def record_trade(self, pnl: float, is_win: bool):
        """Record a completed trade."""
        self.trade_history.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'is_win': is_win
        })

        # Update consecutive losses
        if is_win:
            self.state.consecutive_losses = 0
            self.wins += 1
            self.avg_win = (self.avg_win * (self.wins - 1) + pnl) / self.wins if pnl > 0 else self.avg_win
        else:
            self.state.consecutive_losses += 1
            self.losses += 1
            self.avg_loss = (self.avg_loss * (self.losses - 1) + abs(pnl)) / self.losses if pnl < 0 else self.avg_loss

        self._check_risk_limits()

    def on_position_opened(self, notional: float):
        """
        Update exposure and position count when a position is opened.

        Args:
            notional: The notional value of the position (size * price)
        """
        self.state.open_positions += 1
        self.state.total_exposure += notional
        logger.debug(
            f"Position opened: {self.state.open_positions} positions, "
            f"${self.state.total_exposure:.2f} total exposure"
        )

    def on_position_closed(self, notional: float):
        """
        Update exposure and position count when a position is closed.

        Args:
            notional: The notional value of the closed position
        """
        self.state.open_positions = max(0, self.state.open_positions - 1)
        self.state.total_exposure = max(0.0, self.state.total_exposure - notional)
        logger.debug(
            f"Position closed: {self.state.open_positions} positions, "
            f"${self.state.total_exposure:.2f} total exposure"
        )

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size based on configured method.
        """
        if not self.state.is_trading_allowed:
            return 0.0

        risk_per_unit = abs(entry_price - stop_loss_price)

        if self.sizing.method == 'fixed':
            size = self.sizing.fixed_size

        elif self.sizing.method == 'percent':
            risk_amount = self.current_balance * self.sizing.percent_risk
            size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

        elif self.sizing.method == 'kelly':
            size = self._kelly_position_size(entry_price, risk_per_unit)

        elif self.sizing.method == 'volatility':
            if volatility is None or volatility == 0:
                volatility = 0.01
            target_risk = self.current_balance * self.sizing.percent_risk
            size = target_risk / (volatility * entry_price)

        else:
            size = self.sizing.fixed_size

        # Apply limits
        max_size = self._calculate_max_position_size(entry_price)
        size = min(size, max_size)

        return max(0, size)

    def _kelly_position_size(self, entry_price: float, risk_per_unit: float) -> float:
        """Calculate position size using Kelly Criterion."""
        total_trades = self.wins + self.losses

        if total_trades < 20:
            # Not enough data, use conservative sizing
            return self.current_balance * 0.01 / entry_price

        win_rate = self.wins / total_trades
        avg_win_loss_ratio = self.avg_win / self.avg_loss if self.avg_loss > 0 else 1.0

        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        kelly = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio

        # Apply fraction and constraints
        kelly = max(0, min(kelly * self.sizing.kelly_fraction, 0.25))

        risk_amount = self.current_balance * kelly
        size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

        return size

    def _calculate_max_position_size(self, entry_price: float) -> float:
        """Calculate maximum allowed position size."""
        # Based on position size limit
        max_by_position = self.current_balance * self.limits.max_position_size_pct / entry_price

        # Based on total exposure limit
        remaining_exposure = (
            self.limits.max_total_exposure_pct * self.current_balance -
            self.state.total_exposure
        )
        max_by_exposure = max(0, remaining_exposure / entry_price)

        # Based on leverage
        max_by_leverage = self.current_balance * self.sizing.max_leverage / entry_price

        return min(max_by_position, max_by_exposure, max_by_leverage)

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: Optional[float] = None,
        method: str = 'atr'
    ) -> float:
        """
        Calculate stop loss price.

        Methods:
        - 'atr': Based on Average True Range
        - 'percent': Fixed percentage
        - 'volatility': Based on recent volatility
        """
        if method == 'atr' and atr is not None:
            stop_distance = atr * 2  # 2 ATR stop
        elif method == 'percent':
            stop_distance = entry_price * 0.02  # 2% stop
        else:
            stop_distance = entry_price * 0.02

        if direction == 'long':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        direction: str,
        risk_reward: Optional[float] = None
    ) -> float:
        """Calculate take profit price based on risk/reward ratio."""
        if risk_reward is None:
            risk_reward = self.limits.min_risk_reward

        risk = abs(entry_price - stop_loss_price)
        reward = risk * risk_reward

        if direction == 'long':
            return entry_price + reward
        else:
            return entry_price - reward

    def should_take_trade(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        direction: str
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be taken based on risk parameters.

        Returns:
            Tuple of (should_trade, reason)
        """
        # Check if trading is allowed
        if not self.state.is_trading_allowed:
            return False, f"Trading halted: {self.state.halt_reason}"

        # Check risk/reward ratio
        risk = abs(entry_price - stop_loss_price)
        reward = abs(take_profit_price - entry_price)

        if risk == 0:
            return False, "Invalid stop loss (zero risk)"

        rr_ratio = reward / risk
        if rr_ratio < self.limits.min_risk_reward:
            return False, f"Risk/reward ratio {rr_ratio:.2f} below minimum {self.limits.min_risk_reward}"

        # Check position size
        position_size = self.calculate_position_size(entry_price, stop_loss_price)
        if position_size <= 0:
            return False, "Position size too small or limits exceeded"

        # Check max positions
        if self.state.open_positions >= self.limits.max_open_positions:
            return False, f"Maximum open positions ({self.limits.max_open_positions}) reached"

        return True, "Trade approved"

    def _check_risk_limits(self):
        """Check all risk limits and update trading permission."""
        halt_reasons = []

        # Check drawdown limit
        if self.state.current_drawdown >= self.limits.max_drawdown_pct:
            halt_reasons.append(f"Max drawdown exceeded ({self.state.current_drawdown:.1%})")

        # Check daily loss limit
        if self.initial_balance > 0:
            daily_loss_pct = -self.state.daily_pnl / self.initial_balance
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                halt_reasons.append(f"Daily loss limit exceeded ({daily_loss_pct:.1%})")

        # Check weekly loss limit
        if self.week_start_balance > 0:
            weekly_loss_pct = (self.week_start_balance - self.current_balance) / self.week_start_balance
            if weekly_loss_pct >= self.limits.max_weekly_loss_pct:
                halt_reasons.append(f"Weekly loss limit exceeded ({weekly_loss_pct:.1%})")

        # Check consecutive losses
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            halt_reasons.append(f"Max consecutive losses ({self.state.consecutive_losses})")

        # Update state
        if halt_reasons:
            self.state.is_trading_allowed = False
            self.state.halt_reason = "; ".join(halt_reasons)
            logger.warning(f"Trading halted: {self.state.halt_reason}")
        else:
            self.state.is_trading_allowed = True
            self.state.halt_reason = None

    def _handle_date_rollover(self, new_date):
        """Handle daily/weekly reset."""
        # Store daily PnL
        self.daily_pnl_history.append({
            'date': self.last_reset_date,
            'pnl': self.state.daily_pnl
        })

        # Reset daily PnL
        self.state.daily_pnl = 0.0

        # Check for week rollover (Monday)
        if new_date.weekday() == 0:
            self.week_start_balance = self.current_balance
            self.state.weekly_pnl = 0.0

        self.last_reset_date = new_date

        # Re-enable trading on new day (if only daily limit was hit)
        if 'Daily' in str(self.state.halt_reason):
            self._check_risk_limits()

    def get_risk_report(self) -> Dict:
        """Generate risk report."""
        total_trades = self.wins + self.losses
        return {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'peak_equity': self.peak_equity,
            'current_drawdown': self.state.current_drawdown,
            'max_drawdown_limit': self.limits.max_drawdown_pct,
            'daily_pnl': self.state.daily_pnl,
            'daily_pnl_pct': self.state.daily_pnl / self.initial_balance if self.initial_balance > 0 else 0.0,
            'consecutive_losses': self.state.consecutive_losses,
            'is_trading_allowed': self.state.is_trading_allowed,
            'halt_reason': self.state.halt_reason,
            'win_rate': self.wins / total_trades if total_trades > 0 else 0.0,
            'total_trades': total_trades,
            'position_sizing_method': self.sizing.method,
            'open_positions': self.state.open_positions,
            'max_open_positions': self.limits.max_open_positions,
            'total_exposure': self.state.total_exposure,
            'max_exposure_pct': self.limits.max_total_exposure_pct
        }

    def reset(self):
        """Reset risk manager to initial state."""
        self.current_balance = self.initial_balance
        self.peak_equity = self.initial_balance
        self.state = RiskState()
        self.wins = 0
        self.losses = 0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.equity_history.clear()
        self.trade_history.clear()
        self.daily_pnl_history.clear()


class DynamicRiskManager(RiskManager):
    """
    Extended risk manager with dynamic adjustments based on market conditions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.volatility_history: deque = deque(maxlen=100)
        self.regime = 'normal'

    def update_market_conditions(self, volatility: float):
        """Update market conditions for dynamic adjustment."""
        self.volatility_history.append(volatility)

        # Determine regime
        avg_volatility = np.mean(self.volatility_history) if self.volatility_history else volatility

        if volatility > avg_volatility * 1.5:
            self.regime = 'high_volatility'
        elif volatility < avg_volatility * 0.5:
            self.regime = 'low_volatility'
        else:
            self.regime = 'normal'

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calculate position size with dynamic adjustment."""
        base_size = super().calculate_position_size(entry_price, stop_loss_price, volatility)

        # Adjust based on regime
        if self.regime == 'high_volatility':
            adjustment = 0.5  # Reduce size in high volatility
        elif self.regime == 'low_volatility':
            adjustment = 1.2  # Slightly increase in low volatility
        else:
            adjustment = 1.0

        # Adjust based on recent performance
        if self.state.consecutive_losses >= 3:
            adjustment *= 0.7  # Reduce after losses

        adjusted_size = base_size * adjustment

        # Still apply max limits
        max_size = self._calculate_max_position_size(entry_price)
        return min(adjusted_size, max_size)
