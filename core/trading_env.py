"""
Leap Trading System - Trading Environment
Gymnasium-compatible environment for reinforcement learning with historical data.
"""

import numpy as np
from typing import ClassVar, Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from gymnasium import spaces
import logging

from core.trading_types import Action, EnvConfig, Position, TradingState
from core.trading_env_base import BaseTradingEnvironment

if TYPE_CHECKING:
    from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = ['Action', 'Position', 'TradingState', 'TradingEnvironment', 'MultiSymbolTradingEnv']


class TradingEnvironment(BaseTradingEnvironment):
    """
    Trading environment for backtesting with historical data.

    Features:
    - Multiple position management
    - Risk management (stop loss, take profit, trailing stops)
    - Realistic transaction costs (spread, commission, slippage)
    - Detailed state representation
    """

    metadata: ClassVar[Dict[str, List[str]]] = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        data: np.ndarray,
        features: Optional[np.ndarray] = None,
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
        Initialize backtesting trading environment.

        Args:
            data: OHLCV data array (n_steps, n_columns)
            features: Optional additional features array (n_steps, n_features)
            config: Optional EnvConfig dataclass
            initial_balance: Starting account balance
            commission: Commission rate per trade
            spread: Bid/ask spread
            slippage: Slippage per trade
            leverage: Account leverage
            max_position_size: Max position size as fraction of balance
            stop_loss_pct: Default stop loss percentage
            take_profit_pct: Default take profit percentage
            window_size: Observation window size
            render_mode: Rendering mode
            risk_manager: Optional risk manager
        """
        # Initialize base class
        super().__init__(
            config=config,
            initial_balance=initial_balance,
            commission=commission,
            spread=spread,
            slippage=slippage,
            leverage=leverage,
            max_position_size=max_position_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            window_size=window_size,
            render_mode=render_mode,
            risk_manager=risk_manager
        )

        self.data = data  # OHLCV data: (n_steps, n_columns)
        self.features = features  # Additional features: (n_steps, n_features)

        # Validate window_size against data length
        if data is not None and self.window_size >= len(data):
            raise ValueError(
                f"window_size ({self.window_size}) must be less than data length ({len(data)})"
            )

        # Calculate observation dimension from actual data shape
        self.n_price_features = data.shape[1] if data is not None else 5
        self.n_additional_features = features.shape[1] if features is not None else 0
        self.n_account_features = 8

        obs_dim = (
            self.window_size * (self.n_price_features + self.n_additional_features) +
            self.n_account_features
        )

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
        _ = options  # Required by Gymnasium API but currently unused

        self.current_step = self.window_size
        self.state = TradingState(
            balance=self.initial_balance,
            equity=self.initial_balance,
            peak_equity=self.initial_balance
        )

        self._reset_history()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Get current price
        current_price = self._get_current_price()
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
            self._check_termination()
        )
        truncated = False

        # Record history
        self._record_history(action, reward, current_price)

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------------------
    # Abstract method implementations
    # -------------------------------------------------------------------------

    def _get_current_price(self) -> float:
        """Get current close price from data array."""
        return self.data[self.current_step, 3]  # Close price

    def _get_market_observation(self) -> np.ndarray:
        """Get market observation from data arrays."""
        # Guard against out-of-bounds access after episode termination
        max_step = len(self.data) - 1
        if self.current_step > max_step:
            self.current_step = max_step

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

        return market_obs

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

        # Notify risk manager of position opened
        notional = size * entry_price
        if self.risk_manager is not None:
            self.risk_manager.on_position_opened(notional)

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

        # Notify risk manager of position closed
        notional = position.size * position.entry_price
        if self.risk_manager is not None:
            self.risk_manager.on_position_closed(notional)

    def _close_all_positions(self, price: float):
        """Close all open positions."""
        for position in list(self.state.positions):
            self._close_position(position, price)

    def _get_open_positions(self) -> List[Position]:
        """Get list of open positions."""
        return self.state.positions

    def _has_position(self, position_type: str) -> bool:
        """Check if we have a position of given type."""
        return any(p.type == position_type for p in self.state.positions)

    def _get_unrealized_pnl(self, price: float) -> float:
        """Calculate unrealized PnL at current price."""
        unrealized_pnl = 0.0
        for position in self.state.positions:
            if position.type == 'long':
                unrealized_pnl += (price - position.entry_price) * position.size
            else:
                unrealized_pnl += (position.entry_price - price) * position.size
        return unrealized_pnl

    # -------------------------------------------------------------------------
    # Environment-specific methods
    # -------------------------------------------------------------------------

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
        unrealized_pnl = self._get_unrealized_pnl(current_price)
        self.state.equity = self.state.balance + unrealized_pnl

        # Update peak equity and drawdown
        self._update_drawdown()

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        market_obs = self._get_market_observation()
        account_obs = self._get_account_observation()

        # Combine observations
        obs = np.concatenate([market_obs, account_obs]).astype(np.float32)

        return obs

    def _get_info(self) -> Dict:
        """Get current info dictionary."""
        return self._get_base_info()

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
        stats = super().get_episode_stats()

        # Add backtesting-specific stats
        if 'error' not in stats:
            stats['avg_trade_duration'] = self._calculate_avg_trade_duration()

        return stats

    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration."""
        # This would be tracked properly in production
        return 0.0


class MultiSymbolTradingEnv:
    """
    Environment for trading multiple symbols simultaneously.
    Uses composition to manage multiple TradingEnvironment instances.
    """

    def __init__(
        self,
        symbol_data: Dict[str, np.ndarray],
        symbol_features: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[EnvConfig] = None,
        **kwargs
    ):
        """
        Initialize multi-symbol trading environment.

        Args:
            symbol_data: Dict mapping symbol names to OHLCV data arrays
            symbol_features: Optional dict mapping symbols to feature arrays
            config: Optional EnvConfig for all child environments
            **kwargs: Additional arguments passed to TradingEnvironment
        """
        self.symbols = list(symbol_data.keys())
        self.envs = {
            symbol: TradingEnvironment(
                data=symbol_data[symbol],
                features=symbol_features.get(symbol) if symbol_features else None,
                config=config,
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
        """Reset all environments."""
        obs_list = []
        info_dict = {}

        for symbol, env in self.envs.items():
            obs, info = env.reset(seed=seed)
            obs_list.append(obs)
            info_dict[symbol] = info

        return np.concatenate(obs_list), info_dict

    def step(self, actions):
        """Step all environments with corresponding actions."""
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

    def get_episode_stats(self) -> Dict[str, Dict]:
        """Get episode statistics for all symbols."""
        return {symbol: env.get_episode_stats() for symbol, env in self.envs.items()}
