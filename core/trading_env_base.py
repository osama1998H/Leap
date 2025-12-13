"""
Leap Trading System - Base Trading Environment
Abstract base class with shared functionality for all trading environments.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import gymnasium as gym
from gymnasium import spaces
import logging

from core.trading_types import Action, EnvConfig, Position, TradingState
from evaluation.metrics import MetricsCalculator

if TYPE_CHECKING:
    from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class BaseTradingEnvironment(gym.Env, ABC):
    """
    Abstract base class for trading environments.

    Provides shared functionality for reward calculation, statistics,
    and observation building. Subclasses implement data source specifics.
    """

    metadata: ClassVar[Dict[str, List[str]]] = {'render_modes': ['human', 'log']}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        initial_balance: float = 10000.0,
        commission: float = 0.0001,
        spread: float = 0.0002,
        slippage: float = 0.0001,
        leverage: int = 100,
        max_position_size: float = 0.1,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        window_size: int = 60,
        render_mode: Optional[str] = None,
        risk_manager: Optional['RiskManager'] = None
    ):
        """
        Initialize base trading environment.

        Args:
            config: Optional EnvConfig dataclass (overrides individual params).
                    If provided, individual params are ignored.
            initial_balance: Starting account balance
            commission: Commission rate per trade
            spread: Bid/ask spread
            slippage: Slippage per trade
            leverage: Account leverage
            max_position_size: Max position size as fraction of balance
            stop_loss_pct: Default stop loss percentage
            take_profit_pct: Default take profit percentage
            window_size: Observation window size
            render_mode: Rendering mode ('human', 'log', None)
            risk_manager: Optional risk manager
        """
        super().__init__()

        # Use config if provided, otherwise create from individual params
        # EnvConfig.from_params() handles defaults centrally
        if config is None:
            config = EnvConfig.from_params(
                initial_balance=initial_balance,
                commission=commission,
                spread=spread,
                slippage=slippage,
                leverage=leverage,
                max_position_size=max_position_size,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                window_size=window_size
            )

        # Store config and extract values (single source of truth)
        self._config = config
        self.initial_balance = config.initial_balance
        self.commission = config.commission
        self.spread = config.spread
        self.slippage = config.slippage
        self.leverage = config.leverage
        self.max_position_size = config.max_position_size

        # Validate leverage to prevent division by zero in margin calculations
        if self.leverage <= 0:
            raise ValueError(
                f"leverage must be > 0, got {self.leverage}. "
                f"Leverage is used for margin calculations and cannot be zero or negative."
            )
        self.stop_loss_pct = config.stop_loss_pct
        self.take_profit_pct = config.take_profit_pct
        self.window_size = config.window_size
        self.max_drawdown_threshold = config.max_drawdown_threshold
        self.max_episode_steps = config.max_episode_steps

        self.render_mode = render_mode
        self.risk_manager = risk_manager

        # Metrics calculator for consistent metric calculations
        # Uses 252 trading days for daily data (hourly would be 252*24)
        self._metrics_calculator = MetricsCalculator(
            risk_free_rate=0.02,
            periods_per_year=252  # Can be overridden by subclasses if needed
        )

        # Action space: [HOLD, BUY, SELL, CLOSE]
        self.action_space = spaces.Discrete(4)

        # State and step counter (initialized by subclass reset)
        self.state: TradingState = None
        self.current_step = 0
        self._episode_step = 0  # Steps within current episode (for max_episode_steps)

        # Track previous drawdown for delta-based reward calculation
        # This enables penalizing drawdown INCREASES rather than cumulative drawdown
        self._prev_drawdown = 0.0

        # History tracking
        self.history = self._create_empty_history()

    def _create_empty_history(self) -> Dict[str, List]:
        """Create empty history dictionary."""
        return {
            'balance': [],
            'equity': [],
            'actions': [],
            'rewards': [],
            'positions': [],
            'prices': []
        }

    def _reset_history(self):
        """Reset history with initial values."""
        self.history = {
            'balance': [self.state.balance],
            'equity': [self.state.equity],
            'actions': [],
            'rewards': [],
            'positions': [],
            'prices': []
        }
        # Reset drawdown tracking for new episode
        self._prev_drawdown = 0.0

    # -------------------------------------------------------------------------
    # Abstract methods - must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def _get_current_price(self) -> float:
        """Get current market price."""
        pass

    @abstractmethod
    def _get_market_observation(self) -> np.ndarray:
        """Get market observation (price window + features)."""
        pass

    @abstractmethod
    def _open_position(self, direction: str, price: float):
        """Open a new position."""
        pass

    @abstractmethod
    def _close_position(self, position: Position, price: float):
        """Close a specific position."""
        pass

    @abstractmethod
    def _close_all_positions(self, price: float):
        """Close all open positions."""
        pass

    @abstractmethod
    def _get_open_positions(self) -> List[Position]:
        """Get list of open positions."""
        pass

    @abstractmethod
    def _has_position(self, direction: str) -> bool:
        """Check if position exists in given direction."""
        pass

    @abstractmethod
    def _get_unrealized_pnl(self, price: float) -> float:
        """Calculate unrealized PnL at current price."""
        pass

    # -------------------------------------------------------------------------
    # Shared implementations
    # -------------------------------------------------------------------------

    def _execute_action(self, action: int, price: float):
        """Execute trading action."""
        if action == Action.HOLD:
            return

        elif action == Action.BUY:
            if not self._has_position('long'):
                self._open_position('long', price)

        elif action == Action.SELL:
            if not self._has_position('short'):
                self._open_position('short', price)

        elif action == Action.CLOSE:
            self._close_all_positions(price)

    def _calculate_reward(self, prev_equity: float) -> float:
        """
        Calculate reward using multiple components:
        - Returns-based reward (primary signal)
        - Delta-based drawdown penalty (only penalize INCREASES in drawdown)
        - Small recovery bonus (incentivize reducing drawdown)
        - Position holding cost

        The key insight is that max_drawdown is monotonically increasing within
        an episode, so using it directly as a penalty makes ALL rewards negative
        over time. Instead, we use the CHANGE in drawdown (delta) to only
        penalize when risk increases, while rewarding recovery.

        FIX Critical #3 (v2): Balanced scaling factors for numerical stability
        while maintaining sufficient learning signal.
        - Original: 100x (caused gradient explosion)
        - v1 fix: 10x (too weak, agent didn't learn)
        - v2 fix: 50x with ±5 clipping (balanced)
        """
        # Guard against zero or negative prev_equity (account wiped out)
        if prev_equity <= 0:
            return -1.0

        # === Return Component (primary reward signal) ===
        safe_prev_equity = max(prev_equity, 1e-8)
        returns = (self.state.equity - prev_equity) / safe_prev_equity

        # FIX v2: Increased from 10x to 50x for stronger learning signal
        # 50x provides meaningful gradients while staying bounded
        return_reward = returns * 50.0

        # === Delta-Based Drawdown Penalty ===
        # Calculate current drawdown (resets to 0 when equity makes new high)
        current_drawdown = 0.0
        if self.state.peak_equity > 0:
            current_drawdown = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
            current_drawdown = max(0.0, current_drawdown)  # Ensure non-negative

        # Only penalize when drawdown INCREASES (delta > 0)
        # This avoids the cumulative penalty problem
        drawdown_delta = current_drawdown - self._prev_drawdown
        drawdown_penalty = 0.0
        recovery_bonus = 0.0

        if drawdown_delta > 0:
            # FIX v2: Increased from 5x to 25x for stronger signal
            drawdown_penalty = -drawdown_delta * 25.0
        elif drawdown_delta < 0:
            # FIX v2: Increased from 2.5x to 12.5x
            recovery_bonus = -drawdown_delta * 12.5

        # Update previous drawdown for next step
        self._prev_drawdown = current_drawdown

        # === Position Holding Cost ===
        # Small cost to encourage decisive actions (not holding forever)
        holding_cost = -len(self._get_open_positions()) * 0.001

        # === Combine Reward Components ===
        reward = return_reward + drawdown_penalty + recovery_bonus + holding_cost

        # FIX v2: Clip at ±5 instead of ±10 for tighter bounds
        # Combined with 50x scaling, this gives good signal with stability
        reward = max(-5.0, min(5.0, reward))

        return float(reward)

    def _update_drawdown(self):
        """Update peak equity and max drawdown."""
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

        if self.state.peak_equity > 0:
            drawdown = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
            if drawdown > self.state.max_drawdown:
                self.state.max_drawdown = drawdown

    def _record_history(self, action: int, reward: float, price: float):
        """Record step in history."""
        self.history['balance'].append(self.state.balance)
        self.history['equity'].append(self.state.equity)
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['prices'].append(price)
        self.history['positions'].append(len(self._get_open_positions()))

    def _get_account_observation(self) -> np.ndarray:
        """Get account state observation."""
        current_price = self._get_current_price()
        unrealized_pnl = self._get_unrealized_pnl(current_price)

        return np.array([
            self.state.balance / self.initial_balance,
            self.state.equity / self.initial_balance,
            len(self._get_open_positions()),
            1.0 if self._has_position('long') else 0.0,
            1.0 if self._has_position('short') else 0.0,
            unrealized_pnl / self.initial_balance,
            self.state.max_drawdown,
            self.state.total_pnl / self.initial_balance
        ])

    def _get_base_info(self) -> Dict[str, Any]:
        """Get common info dictionary fields."""
        return {
            'balance': self.state.balance,
            'equity': self.state.equity,
            'total_trades': self.state.total_trades,
            'winning_trades': self.state.winning_trades,
            'losing_trades': self.state.losing_trades,
            'win_rate': self.state.winning_trades / max(1, self.state.total_trades),
            'total_pnl': self.state.total_pnl,
            'max_drawdown': self.state.max_drawdown,
            'current_step': self.current_step,
            'open_positions': len(self._get_open_positions())
        }

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        if self.state.equity <= 0:
            return True
        if self.state.max_drawdown >= self.max_drawdown_threshold:
            return True
        return False

    # -------------------------------------------------------------------------
    # Statistics methods
    # -------------------------------------------------------------------------

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe ratio using centralized MetricsCalculator.

        Delegates to MetricsCalculator for consistent metric calculations
        across the codebase. Risk-free rate is configured at initialization.
        """
        return self._metrics_calculator.sharpe_ratio(returns)

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sortino ratio using centralized MetricsCalculator.

        Delegates to MetricsCalculator for consistent metric calculations
        across the codebase. Risk-free rate is configured at initialization.
        """
        return self._metrics_calculator.sortino_ratio(returns)

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if self.state.gross_loss == 0:
            return float('inf') if self.state.gross_profit > 0 else 0.0

        return self.state.gross_profit / self.state.gross_loss

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get comprehensive episode statistics."""
        equity_curve = np.array(self.history['equity'])

        if len(equity_curve) < 2:
            return {'error': 'Not enough data'}

        returns = np.diff(equity_curve) / equity_curve[:-1]

        stats = {
            'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0],
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self.state.max_drawdown,
            'win_rate': self.state.winning_trades / max(1, self.state.total_trades),
            'profit_factor': self._calculate_profit_factor(),
            'total_trades': self.state.total_trades,
            'final_equity': self.state.equity
        }

        return stats
