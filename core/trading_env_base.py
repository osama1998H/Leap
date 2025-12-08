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
            config: Optional EnvConfig dataclass (overrides individual params)
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

        # Use config if provided, otherwise use individual params
        if config is not None:
            self.initial_balance = config.initial_balance
            self.commission = config.commission
            self.spread = config.spread
            self.slippage = config.slippage
            self.leverage = config.leverage
            self.max_position_size = config.max_position_size
            self.stop_loss_pct = config.stop_loss_pct
            self.take_profit_pct = config.take_profit_pct
            self.window_size = config.window_size
            self.max_drawdown_threshold = config.max_drawdown_threshold
        else:
            self.initial_balance = initial_balance
            self.commission = commission
            self.spread = spread
            self.slippage = slippage
            self.leverage = leverage
            self.max_position_size = max_position_size
            self.stop_loss_pct = stop_loss_pct
            self.take_profit_pct = take_profit_pct
            self.window_size = window_size
            self.max_drawdown_threshold = 0.5

        self.render_mode = render_mode
        self.risk_manager = risk_manager

        # Action space: [HOLD, BUY, SELL, CLOSE]
        self.action_space = spaces.Discrete(4)

        # State and step counter (initialized by subclass reset)
        self.state: TradingState = None
        self.current_step = 0

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
        - Returns-based reward
        - Risk-adjusted reward
        - Drawdown penalty
        """
        # Guard against zero or negative prev_equity (account wiped out)
        if prev_equity <= 0:
            return -1.0

        # Return component
        safe_prev_equity = max(prev_equity, 1e-8)
        returns = (self.state.equity - prev_equity) / safe_prev_equity
        return_reward = returns * 100  # Scale up

        # Drawdown penalty
        drawdown_penalty = -self.state.max_drawdown * 10

        # Position holding cost (encourage decisive actions)
        holding_cost = -len(self._get_open_positions()) * 0.0001

        # Combine rewards
        reward = return_reward + drawdown_penalty + holding_cost

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
            'positions': len(self._get_open_positions())
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

    def _calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_sortino_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)

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
