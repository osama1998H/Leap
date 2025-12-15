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

        # === New v2 Reward Shaping Parameters ===
        self.tc_penalty_scale = config.tc_penalty_scale
        self.dd_threshold = config.dd_threshold
        self.dd_penalty_scale = config.dd_penalty_scale
        self.vol_penalty_scale = config.vol_penalty_scale
        self.vol_window = config.vol_window
        self.reward_clip = config.reward_clip
        self.normalize_reward = config.normalize_reward

        # Legacy parameters (kept for backward compatibility)
        self.return_scale = config.return_scale
        self.drawdown_penalty_scale = config.drawdown_penalty_scale
        self.recovery_bonus_scale = config.recovery_bonus_scale
        self.holding_cost_per_position = config.holding_cost

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

        # === New v2 tracking variables ===
        # Track previous position weight for transaction cost penalty
        self._prev_position_weight = 0.0
        # Rolling buffer for return volatility calculation
        self._return_history: List[float] = []

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
            'prices': [],
            # Reward component tracking for diagnostics (v2)
            'reward_components': {
                'log_return': [],           # Log-return PnL component
                'tc_penalty': [],           # Transaction cost penalty
                'dd_penalty': [],           # Drawdown penalty (threshold-based)
                'vol_penalty': [],          # Volatility penalty
                'raw_reward': [],           # Before clipping
                # Legacy components (for backward compatibility analysis)
                'return_reward': [],
                'drawdown_penalty': [],
                'recovery_bonus': [],
                'holding_cost': [],
            }
        }

    def _reset_history(self):
        """Reset history with initial values."""
        self.history = {
            'balance': [self.state.balance],
            'equity': [self.state.equity],
            'actions': [],
            'rewards': [],
            'positions': [],
            'prices': [],
            # Reward component tracking for diagnostics (v2)
            'reward_components': {
                'log_return': [],
                'tc_penalty': [],
                'dd_penalty': [],
                'vol_penalty': [],
                'raw_reward': [],
                # Legacy components
                'return_reward': [],
                'drawdown_penalty': [],
                'recovery_bonus': [],
                'holding_cost': [],
            }
        }
        # Reset drawdown tracking for new episode
        self._prev_drawdown = 0.0
        # Reset v2 tracking variables
        self._prev_position_weight = 0.0
        self._return_history = []

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
        Calculate reward using the v2 log-return based formula:

        r_t = log(E_t / E_{t-1})                    # Log-return (scale-stable, additive)
            - c_tc * |Δw_t|                         # Transaction cost penalty
            - λ_dd * max(0, DD_t - DD_threshold)    # Threshold-based drawdown penalty
            - λ_vol * σ̂_t²                          # Optional volatility penalty

        Key improvements over v1:
        1. Log-returns are scale-stable and additive over time (better for PPO)
        2. Transaction cost penalty discourages overtrading (position churn)
        3. Threshold-based drawdown only penalizes when exceeding tolerance
        4. Optional volatility penalty discourages high-variance strategies

        All parameters are configurable via EnvConfig for tuning.
        """
        # Guard against zero or negative prev_equity (account wiped out)
        if prev_equity <= 0:
            self._record_reward_components_v2(0.0, 0.0, 0.0, 0.0, -1.0)
            return -1.0

        # === 1. Log-Return Component (primary reward signal) ===
        # Log-returns are scale-stable and additive: log(E_T/E_0) = Σ log(E_t/E_{t-1})
        safe_prev_equity = max(prev_equity, 1e-8)
        safe_curr_equity = max(self.state.equity, 1e-8)
        log_return = np.log(safe_curr_equity / safe_prev_equity)

        # Clip extreme log-returns (e.g., from news gaps) before scaling
        log_return = np.clip(log_return, -0.5, 0.5)  # ±50% max per step

        # Track for volatility calculation
        self._return_history.append(log_return)
        if len(self._return_history) > self.vol_window:
            self._return_history.pop(0)

        # === 2. Transaction Cost Penalty ===
        # Penalize position changes to discourage overtrading
        # Position weight: w = signed_exposure / equity
        current_position_weight = self._get_position_weight()
        position_change = abs(current_position_weight - self._prev_position_weight)
        tc_penalty = -self.tc_penalty_scale * position_change

        # Update for next step
        self._prev_position_weight = current_position_weight

        # === 3. Threshold-Based Drawdown Penalty ===
        # Only penalize when drawdown exceeds tolerance threshold
        current_drawdown = 0.0
        if self.state.peak_equity > 0:
            current_drawdown = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
            current_drawdown = max(0.0, current_drawdown)

        excess_drawdown = max(0.0, current_drawdown - self.dd_threshold)
        dd_penalty = -self.dd_penalty_scale * excess_drawdown

        # Update previous drawdown for tracking (used by some metrics)
        self._prev_drawdown = current_drawdown

        # === 4. Optional Volatility Penalty ===
        # Penalize high-variance strategies (disabled by default)
        vol_penalty = 0.0
        if self.vol_penalty_scale > 0 and len(self._return_history) >= 2:
            return_variance = np.var(self._return_history)
            vol_penalty = -self.vol_penalty_scale * return_variance

        # === Combine Reward Components ===
        raw_reward = log_return + tc_penalty + dd_penalty + vol_penalty

        # Clip reward for numerical stability
        reward = np.clip(raw_reward, -self.reward_clip, self.reward_clip)

        # Track components for diagnostics
        self._record_reward_components_v2(
            log_return, tc_penalty, dd_penalty, vol_penalty, raw_reward
        )

        return float(reward)

    def _get_position_weight(self) -> float:
        """
        Calculate current position weight (exposure ratio).

        Returns:
            w in [-1, 1]: -1 = full short, 0 = flat, +1 = full long
            Scaled by position value relative to equity.
        """
        positions = self._get_open_positions()
        if not positions or self.state.equity <= 0:
            return 0.0

        # Calculate net signed exposure
        net_exposure = 0.0
        for pos in positions:
            # Position value = size * entry_price (approximate current value)
            pos_value = pos.size * pos.entry_price
            if pos.type == 'long':
                net_exposure += pos_value
            else:  # short
                net_exposure -= pos_value

        # Normalize by equity to get weight in [-1, 1] range
        # Cap at ±1 even with leverage
        weight = net_exposure / self.state.equity
        return np.clip(weight, -1.0, 1.0)

    def _record_reward_components_v2(
        self,
        log_return: float,
        tc_penalty: float,
        dd_penalty: float,
        vol_penalty: float,
        raw_reward: float
    ):
        """Record v2 reward components for analysis."""
        if 'reward_components' in self.history:
            # New v2 components
            self.history['reward_components']['log_return'].append(log_return)
            self.history['reward_components']['tc_penalty'].append(tc_penalty)
            self.history['reward_components']['dd_penalty'].append(dd_penalty)
            self.history['reward_components']['vol_penalty'].append(vol_penalty)
            self.history['reward_components']['raw_reward'].append(raw_reward)

            # Legacy components (zeros for backward compatibility with analysis tools)
            self.history['reward_components']['return_reward'].append(log_return)
            self.history['reward_components']['drawdown_penalty'].append(dd_penalty)
            self.history['reward_components']['recovery_bonus'].append(0.0)
            self.history['reward_components']['holding_cost'].append(tc_penalty)

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

        # Add reward component statistics if available
        reward_stats = self.get_reward_component_stats()
        if reward_stats:
            stats['reward_components'] = reward_stats

        return stats

    def get_reward_component_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics for reward components to diagnose reward signal issues.

        Returns mean, std, min, max for each reward component to identify
        which components dominate the reward signal.
        """
        if 'reward_components' not in self.history:
            return None

        components = self.history['reward_components']
        stats = {}

        for name, values in components.items():
            if len(values) == 0:
                continue

            arr = np.array(values)
            stats[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'sum': float(np.sum(arr)),
                'nonzero_pct': float(np.count_nonzero(arr) / len(arr) * 100)
            }

        return stats if stats else None

    def get_action_distribution(self) -> Dict[str, float]:
        """
        Get distribution of actions taken during the episode.

        Returns percentage of each action type for debugging policy behavior.
        """
        actions = self.history.get('actions', [])
        if not actions:
            return {}

        total = len(actions)
        action_counts = {
            'HOLD': 0,
            'BUY': 0,
            'SELL': 0,
            'CLOSE': 0
        }

        for action in actions:
            if action == Action.HOLD:
                action_counts['HOLD'] += 1
            elif action == Action.BUY:
                action_counts['BUY'] += 1
            elif action == Action.SELL:
                action_counts['SELL'] += 1
            elif action == Action.CLOSE:
                action_counts['CLOSE'] += 1

        # Convert to percentages
        return {k: (v / total * 100) for k, v in action_counts.items()}
