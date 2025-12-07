"""
Leap Trading System - Trading Environment
Gymnasium-compatible environment for reinforcement learning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum
import gymnasium as gym
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


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


class TradingEnvironment(gym.Env):
    """
    Advanced trading environment for reinforcement learning.
    Features:
    - Multiple position management
    - Risk management (stop loss, take profit, trailing stops)
    - Realistic transaction costs (spread, commission, slippage)
    - Detailed state representation
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        data: np.ndarray,
        features: Optional[np.ndarray] = None,
        initial_balance: float = 10000.0,
        commission: float = 0.0001,  # 0.01% per trade
        spread: float = 0.0002,  # 2 pips for forex
        slippage: float = 0.0001,  # 1 pip slippage
        leverage: int = 100,
        max_position_size: float = 0.1,  # Max 10% of balance
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.04,  # 4% take profit
        window_size: int = 60,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.data = data  # OHLCV data: (n_steps, 5)
        self.features = features  # Additional features: (n_steps, n_features)
        self.initial_balance = initial_balance
        self.commission = commission
        self.spread = spread
        self.slippage = slippage
        self.leverage = leverage
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.window_size = window_size
        self.render_mode = render_mode

        # Calculate observation dimension
        self.n_price_features = 5  # OHLCV
        self.n_additional_features = features.shape[1] if features is not None else 0
        self.n_account_features = 8  # balance, equity, position, unrealized_pnl, etc.

        obs_dim = (
            window_size * (self.n_price_features + self.n_additional_features) +
            self.n_account_features
        )

        # Action space: [HOLD, BUY, SELL, CLOSE]
        self.action_space = spaces.Discrete(4)

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.state = TradingState(
            balance=self.initial_balance,
            equity=self.initial_balance,
            peak_equity=self.initial_balance
        )

        self.history = {
            'balance': [self.initial_balance],
            'equity': [self.initial_balance],
            'actions': [],
            'rewards': [],
            'positions': [],
            'prices': []
        }

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Get current price
        current_price = self.data[self.current_step, 3]  # Close price
        prev_equity = self.state.equity

        # Execute action
        self._execute_action(action, current_price)

        # Update positions and calculate PnL
        self._update_positions(current_price)

        # Calculate reward
        reward = self._calculate_reward(prev_equity)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = (
            self.current_step >= len(self.data) - 1 or
            self.state.equity <= 0 or
            self.state.max_drawdown >= 0.5  # 50% max drawdown
        )
        truncated = False

        # Record history
        self.history['balance'].append(self.state.balance)
        self.history['equity'].append(self.state.equity)
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['prices'].append(current_price)

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

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

    def _open_position(self, position_type: str, price: float):
        """Open a new position."""
        # Calculate position size based on risk
        position_value = self.state.balance * self.max_position_size
        size = position_value / price

        # Apply spread and slippage
        if position_type == 'long':
            entry_price = price * (1 + self.spread / 2 + self.slippage)
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:
            entry_price = price * (1 - self.spread / 2 - self.slippage)
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)

        # Commission
        commission_cost = position_value * self.commission
        self.state.balance -= commission_cost

        position = Position(
            type=position_type,
            entry_price=entry_price,
            size=size,
            entry_time=self.current_step,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.state.positions.append(position)
        self.state.total_trades += 1

    def _close_position(self, position: Position, price: float):
        """Close a specific position."""
        # Apply spread and slippage
        if position.type == 'long':
            exit_price = price * (1 - self.spread / 2 - self.slippage)
            pnl = (exit_price - position.entry_price) * position.size
        else:
            exit_price = price * (1 + self.spread / 2 + self.slippage)
            pnl = (position.entry_price - exit_price) * position.size

        # Commission
        position_value = position.size * exit_price
        commission_cost = position_value * self.commission
        pnl -= commission_cost

        # Update state
        self.state.balance += pnl
        self.state.total_pnl += pnl

        # Track gross profit and gross loss for profit factor calculation
        if pnl > 0:
            self.state.winning_trades += 1
            self.state.gross_profit += pnl
        else:
            self.state.losing_trades += 1
            self.state.gross_loss += abs(pnl)

        self.state.positions.remove(position)

    def _close_all_positions(self, price: float):
        """Close all open positions."""
        for position in list(self.state.positions):
            self._close_position(position, price)

    def _update_positions(self, current_price: float):
        """Update positions, check stop loss/take profit."""
        for position in list(self.state.positions):
            # Check stop loss
            if position.type == 'long':
                if current_price <= position.stop_loss:
                    self._close_position(position, position.stop_loss)
                    continue
                elif current_price >= position.take_profit:
                    self._close_position(position, position.take_profit)
                    continue
            else:
                if current_price >= position.stop_loss:
                    self._close_position(position, position.stop_loss)
                    continue
                elif current_price <= position.take_profit:
                    self._close_position(position, position.take_profit)
                    continue

        # Calculate unrealized PnL and equity
        unrealized_pnl = 0.0
        for position in self.state.positions:
            if position.type == 'long':
                unrealized_pnl += (current_price - position.entry_price) * position.size
            else:
                unrealized_pnl += (position.entry_price - current_price) * position.size

        self.state.equity = self.state.balance + unrealized_pnl

        # Update peak equity and drawdown
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

        drawdown = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown

    def _has_position(self, position_type: str) -> bool:
        """Check if we have a position of given type."""
        return any(p.type == position_type for p in self.state.positions)

    def _calculate_reward(self, prev_equity: float) -> float:
        """
        Calculate reward using multiple components:
        - Returns-based reward
        - Risk-adjusted reward
        - Drawdown penalty
        """
        # Return component
        returns = (self.state.equity - prev_equity) / prev_equity
        return_reward = returns * 100  # Scale up

        # Drawdown penalty
        drawdown_penalty = -self.state.max_drawdown * 10

        # Position holding cost (encourage decisive actions)
        holding_cost = -len(self.state.positions) * 0.0001

        # Combine rewards
        reward = return_reward + drawdown_penalty + holding_cost

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Price window
        start_idx = self.current_step - self.window_size
        price_window = self.data[start_idx:self.current_step].flatten()

        # Feature window
        if self.features is not None:
            feature_window = self.features[start_idx:self.current_step].flatten()
            market_obs = np.concatenate([price_window, feature_window])
        else:
            market_obs = price_window

        # Normalize market observations
        market_obs = (market_obs - np.mean(market_obs)) / (np.std(market_obs) + 1e-8)

        # Account state
        current_price = self.data[self.current_step, 3]
        unrealized_pnl = sum(
            (current_price - p.entry_price) * p.size if p.type == 'long'
            else (p.entry_price - current_price) * p.size
            for p in self.state.positions
        )

        account_obs = np.array([
            self.state.balance / self.initial_balance,
            self.state.equity / self.initial_balance,
            len(self.state.positions),
            1.0 if self._has_position('long') else 0.0,
            1.0 if self._has_position('short') else 0.0,
            unrealized_pnl / self.initial_balance,
            self.state.max_drawdown,
            self.state.total_pnl / self.initial_balance
        ])

        # Combine observations
        obs = np.concatenate([market_obs, account_obs]).astype(np.float32)

        return obs

    def _get_info(self) -> Dict:
        """Get current info dictionary."""
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
            'positions': len(self.state.positions)
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"Step: {info['current_step']}, "
                  f"Balance: ${info['balance']:.2f}, "
                  f"Equity: ${info['equity']:.2f}, "
                  f"Trades: {info['total_trades']}, "
                  f"Win Rate: {info['win_rate']:.2%}")

    def get_episode_stats(self) -> Dict:
        """Get comprehensive episode statistics."""
        equity_curve = np.array(self.history['equity'])
        returns = np.diff(equity_curve) / equity_curve[:-1]

        stats = {
            'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0],
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self.state.max_drawdown,
            'win_rate': self.state.winning_trades / max(1, self.state.total_trades),
            'profit_factor': self._calculate_profit_factor(),
            'total_trades': self.state.total_trades,
            'avg_trade_duration': self._calculate_avg_trade_duration(),
            'final_equity': self.state.equity
        }

        return stats

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
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

    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration."""
        # This would be tracked properly in production
        return 0.0


class MultiSymbolTradingEnv(gym.Env):
    """
    Environment for trading multiple symbols simultaneously.
    """

    def __init__(
        self,
        symbol_data: Dict[str, np.ndarray],
        symbol_features: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ):
        self.symbols = list(symbol_data.keys())
        self.envs = {
            symbol: TradingEnvironment(
                data=symbol_data[symbol],
                features=symbol_features.get(symbol) if symbol_features else None,
                **kwargs
            )
            for symbol in self.symbols
        }

        # Combined action and observation spaces
        single_env = self.envs[self.symbols[0]]
        n_symbols = len(self.symbols)

        self.action_space = spaces.MultiDiscrete([4] * n_symbols)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_symbols * single_env.observation_space.shape[0],),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs_list = []
        info_dict = {}

        for symbol, env in self.envs.items():
            obs, info = env.reset(seed=seed)
            obs_list.append(obs)
            info_dict[symbol] = info

        return np.concatenate(obs_list), info_dict

    def step(self, actions):
        obs_list = []
        total_reward = 0.0
        terminated = False
        truncated = False
        info_dict = {}

        for i, (symbol, env) in enumerate(self.envs.items()):
            obs, reward, term, trunc, info = env.step(actions[i])
            obs_list.append(obs)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            info_dict[symbol] = info

        return np.concatenate(obs_list), total_reward, terminated, truncated, info_dict
