"""
Leap Trading System - Live Trading Environment
Extends the base trading environment with real MT5 integration.
"""

import numpy as np
import logging
from typing import Optional, Dict, List, Tuple, Any, ClassVar, TYPE_CHECKING
from gymnasium import spaces

from core.trading_types import Action, EnvConfig, Position, TradingState, LiveTradingState
from core.trading_env_base import BaseTradingEnvironment
from core.mt5_broker import MT5BrokerGateway, OrderType, MT5Position
from core.position_sync import PositionSynchronizer, PositionEvent, PositionChange
from core.order_manager import OrderManager, TradingSignal, SignalType

if TYPE_CHECKING:
    from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class LiveTradingEnvironment(BaseTradingEnvironment):
    """
    Live trading environment with MT5 integration.

    Features:
    - Real-time market data from MT5
    - Real account state synchronization
    - Position synchronization with broker
    - Open status control for position entry
    - Paper mode for safe testing
    - Compatible with PPO agent interface
    """

    metadata: ClassVar[Dict[str, List[str]]] = {'render_modes': ['human', 'log']}

    def __init__(
        self,
        broker: MT5BrokerGateway,
        symbol: str = 'EURUSD',
        data_pipeline=None,
        config: Optional[EnvConfig] = None,
        risk_manager: Optional['RiskManager'] = None,
        initial_balance: float = 10000.0,
        window_size: int = 60,
        max_positions: int = 5,
        default_sl_pips: float = 50.0,
        default_tp_pips: float = 100.0,
        risk_per_trade: float = 0.01,
        render_mode: Optional[str] = None,
        paper_mode: bool = True
    ):
        """
        Initialize live trading environment.

        Args:
            broker: MT5 broker gateway
            symbol: Trading symbol
            data_pipeline: Data pipeline for feature computation
            config: Optional EnvConfig dataclass
            risk_manager: Risk manager for position sizing
            initial_balance: Initial balance (for paper mode)
            window_size: Observation window size
            max_positions: Maximum concurrent positions
            default_sl_pips: Default stop loss in pips
            default_tp_pips: Default take profit in pips
            risk_per_trade: Risk per trade as fraction
            render_mode: Rendering mode
            paper_mode: If True, simulate trades; if False, execute real trades
        """
        # Initialize base class
        super().__init__(
            config=config,
            initial_balance=initial_balance,
            window_size=window_size,
            render_mode=render_mode,
            risk_manager=risk_manager
        )

        self.broker = broker
        self.symbol = symbol
        self.data_pipeline = data_pipeline
        self.max_positions = max_positions
        self.default_sl_pips = default_sl_pips
        self.default_tp_pips = default_tp_pips
        self.risk_per_trade = risk_per_trade
        self.paper_mode = paper_mode

        # Initialize components
        self.position_sync = PositionSynchronizer(
            broker=broker,
            magic_number=broker.magic_number
        )

        self.order_manager = OrderManager(
            broker=broker,
            risk_manager=risk_manager,
            default_sl_pips=default_sl_pips,
            default_tp_pips=default_tp_pips,
            default_risk_percent=risk_per_trade
        )

        # Register position callbacks
        self.position_sync.register_callback(
            PositionEvent.CLOSED, self._on_position_closed
        )
        self.position_sync.register_callback(
            PositionEvent.SL_HIT, self._on_sl_hit
        )
        self.position_sync.register_callback(
            PositionEvent.TP_HIT, self._on_tp_hit
        )

        # Data buffers
        self._price_buffer: List[np.ndarray] = []
        self._feature_buffer: List[np.ndarray] = []
        self._max_buffer_size = window_size + 100

        # Calculate observation dimension
        n_price_features = 5
        n_additional_features = 100  # Estimated, adjusted on first observation
        n_account_features = 12  # Extended account features for live trading

        obs_dim = (
            window_size * (n_price_features + n_additional_features) +
            n_account_features
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Trading state (use LiveTradingState for extended features)
        self.state: LiveTradingState = LiveTradingState(
            balance=initial_balance,
            equity=initial_balance,
            peak_equity=initial_balance,
            session_start_balance=initial_balance
        )

        # Paper trading state
        self._paper_positions: List[Position] = []
        self._paper_balance = initial_balance

        logger.info(f"Live trading environment initialized for {symbol}")

    # -------------------------------------------------------------------------
    # Live trading specific properties and methods
    # -------------------------------------------------------------------------

    @property
    def open_status(self) -> bool:
        """Check if new positions can be opened."""
        return self.state.open_status

    @open_status.setter
    def open_status(self, value: bool):
        """Set open status."""
        self.state.open_status = value
        logger.info(f"Open status set to: {value}")

    def set_open_status(self, can_open: bool):
        """Set whether new positions can be opened."""
        self.open_status = can_open

    def set_close_only(self, close_only: bool):
        """Set close-only mode (no new positions, only closing allowed)."""
        self.state.close_only = close_only
        if close_only:
            self.state.open_status = False
        logger.info(f"Close only mode: {close_only}")

    # -------------------------------------------------------------------------
    # Gymnasium interface
    # -------------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Sync with broker
        if self.broker.is_connected:
            self._sync_with_broker()

        # Reset state
        account = self.broker.get_account_info() if self.broker.is_connected else None

        if account and not self.paper_mode:
            self.state = LiveTradingState(
                balance=account.balance,
                equity=account.equity,
                peak_equity=account.equity,
                real_balance=account.balance,
                real_equity=account.equity,
                real_margin=account.margin,
                real_free_margin=account.free_margin,
                real_margin_level=account.margin_level,
                session_start_balance=account.balance
            )
        else:
            self.state = LiveTradingState(
                balance=self.initial_balance,
                equity=self.initial_balance,
                peak_equity=self.initial_balance,
                session_start_balance=self.initial_balance
            )

        # Clear paper positions
        self._paper_positions.clear()
        self._paper_balance = self.initial_balance

        # Reset buffers
        self._price_buffer.clear()
        self._feature_buffer.clear()

        # Fetch initial data
        self._fetch_initial_data()

        # Reset history
        self._reset_history()

        self.current_step = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        prev_equity = self.state.equity

        # Sync positions
        if self.broker.is_connected:
            self.position_sync.sync()

        # Get current price for action execution
        price = self._get_current_price()

        # Execute action with open_status check for live trading
        # BUY/SELL actions require open_status=True and close_only=False
        if action in [Action.BUY, Action.SELL]:
            if self.state.open_status and not self.state.close_only:
                self._execute_action(action, price)
        else:
            # HOLD and CLOSE actions are always allowed
            self._execute_action(action, price)

        # Update state from broker
        self._sync_with_broker()

        # Calculate reward
        reward = self._calculate_reward(prev_equity)

        # Update step
        self.current_step += 1

        # Check termination conditions
        terminated = self._check_termination()
        truncated = False

        # Record history
        tick = self.broker.get_current_tick(self.symbol)
        price = tick.bid if tick else 0.0
        self._record_history(action, reward, price)

        # Get observation
        self._update_data_buffer()
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------------------
    # Abstract method implementations
    # -------------------------------------------------------------------------

    def _get_current_price(self) -> float:
        """Get current price from broker."""
        tick = self.broker.get_current_tick(self.symbol)
        return tick.bid if tick else 0.0

    def _get_market_observation(self) -> np.ndarray:
        """Get market observation from price/feature buffers."""
        # Price window
        if len(self._price_buffer) >= self.window_size:
            price_window = np.array(self._price_buffer[-self.window_size:]).flatten()
        else:
            # Pad with zeros if not enough data
            padding_size = self.window_size - len(self._price_buffer)
            if self._price_buffer:
                prices = np.array(self._price_buffer).flatten()
                price_window = np.pad(prices, (padding_size * 5, 0), mode='edge')
            else:
                price_window = np.zeros(self.window_size * 5)

        # Feature window
        if self._feature_buffer and len(self._feature_buffer) >= self.window_size:
            feature_window = np.array(self._feature_buffer[-self.window_size:]).flatten()
        else:
            n_features = 100  # Estimated
            feature_window = np.zeros(self.window_size * n_features)

        # Normalize market observations
        market_obs = np.concatenate([price_window, feature_window])
        market_mean = np.mean(market_obs)
        market_std = np.std(market_obs)
        if market_std > 0:
            market_obs = (market_obs - market_mean) / market_std
        market_obs = np.nan_to_num(market_obs, nan=0.0, posinf=0.0, neginf=0.0)

        return market_obs

    def _open_position(self, direction: str, price: float):
        """Open a new position."""
        # Check position limit
        current_positions = len(self._get_open_positions())
        if current_positions >= self.max_positions:
            logger.info(f"Max positions ({self.max_positions}) reached")
            return

        signal = TradingSignal(
            signal_type=SignalType.BUY if direction == 'long' else SignalType.SELL,
            symbol=self.symbol,
            stop_loss_pips=self.default_sl_pips,
            take_profit_pips=self.default_tp_pips,
            risk_percent=self.risk_per_trade,
            source="live_env"
        )

        if self.paper_mode:
            self._execute_paper_trade(signal)
        else:
            execution = self.order_manager.execute_signal(signal)
            if execution.executed:
                self.state.total_trades += 1
                self.state.session_trades += 1

    def _close_position(self, position: Position, price: float):
        """Close a specific position."""
        # In live mode, use order manager
        if not self.paper_mode:
            # Position closing is handled by position_sync
            pass
        else:
            self._close_paper_position(position, price)

    def _close_all_positions(self, price: float):
        """Close all open positions."""
        if self.paper_mode:
            self._close_all_paper_positions()
        else:
            signal = TradingSignal(
                signal_type=SignalType.CLOSE,
                symbol=self.symbol,
                source="live_env"
            )
            self.order_manager.execute_signal(signal)

    def _get_open_positions(self) -> List[Position]:
        """Get list of open positions."""
        if self.paper_mode:
            return self._paper_positions
        else:
            return self.position_sync.get_positions(self.symbol)

    def _has_position(self, direction: str) -> bool:
        """Check if we have a position in given direction."""
        if self.paper_mode:
            return any(p.type == direction for p in self._paper_positions)
        else:
            return self.position_sync.has_position(self.symbol, direction)

    def _get_unrealized_pnl(self, price: float) -> float:
        """Calculate unrealized PnL."""
        if self.paper_mode:
            return self._calculate_paper_unrealized_pnl()
        else:
            return self.position_sync.get_unrealized_pnl(self.symbol)

    # -------------------------------------------------------------------------
    # Paper trading methods
    # -------------------------------------------------------------------------

    def _execute_paper_trade(self, signal: TradingSignal):
        """Execute a paper trade (simulated)."""
        tick = self.broker.get_current_tick(self.symbol)
        if tick is None:
            return

        symbol_info = self.broker.get_symbol_info(self.symbol)
        if symbol_info is None:
            return

        # Calculate position parameters
        pip_size = symbol_info.point * 10

        if signal.signal_type == SignalType.BUY:
            entry_price = tick.ask
            sl = entry_price - (self.default_sl_pips * pip_size)
            tp = entry_price + (self.default_tp_pips * pip_size)
            pos_type = 'long'
        else:
            entry_price = tick.bid
            sl = entry_price + (self.default_sl_pips * pip_size)
            tp = entry_price - (self.default_tp_pips * pip_size)
            pos_type = 'short'

        # Calculate volume
        risk_amount = self._paper_balance * self.risk_per_trade
        try:
            pip_value = symbol_info.trade_tick_value * (pip_size / symbol_info.trade_tick_size)
        except Exception:
            pip_value = 10.0  # Fallback for major pairs
        volume = risk_amount / (self.default_sl_pips * pip_value)
        volume = max(0.01, min(volume, 10.0))

        position = Position(
            type=pos_type,
            entry_price=entry_price,
            size=volume,
            entry_time=self.current_step,
            stop_loss=sl,
            take_profit=tp
        )

        self._paper_positions.append(position)
        self.state.total_trades += 1
        self.state.session_trades += 1

        logger.info(f"Paper trade opened: {pos_type} {volume} @ {entry_price}")

    def _close_all_paper_positions(self):
        """Close all paper positions."""
        tick = self.broker.get_current_tick(self.symbol)
        if tick is None:
            return

        symbol_info = self.broker.get_symbol_info(self.symbol)
        contract_size = symbol_info.trade_contract_size if symbol_info else 100000

        for position in list(self._paper_positions):
            if position.type == 'long':
                exit_price = tick.bid
                pnl = (exit_price - position.entry_price) * position.size * contract_size
            else:
                exit_price = tick.ask
                pnl = (position.entry_price - exit_price) * position.size * contract_size

            self._paper_balance += pnl
            self.state.total_pnl += pnl

            if pnl > 0:
                self.state.winning_trades += 1
                self.state.gross_profit += pnl
            else:
                self.state.losing_trades += 1
                self.state.gross_loss += abs(pnl)

            self._paper_positions.remove(position)
            logger.info(f"Paper position closed: PnL = {pnl:.2f}")

    def _close_paper_position(self, position: Position, exit_price: float):
        """Close a specific paper position."""
        symbol_info = self.broker.get_symbol_info(self.symbol)
        contract_size = symbol_info.trade_contract_size if symbol_info else 100000

        if position.type == 'long':
            pnl = (exit_price - position.entry_price) * position.size * contract_size
        else:
            pnl = (position.entry_price - exit_price) * position.size * contract_size

        self._paper_balance += pnl
        self.state.total_pnl += pnl

        if pnl > 0:
            self.state.winning_trades += 1
            self.state.gross_profit += pnl
        else:
            self.state.losing_trades += 1
            self.state.gross_loss += abs(pnl)

        self._paper_positions.remove(position)

    def _update_paper_positions(self):
        """Update paper positions with current prices (check SL/TP)."""
        tick = self.broker.get_current_tick(self.symbol)
        if tick is None:
            return

        for position in list(self._paper_positions):
            if position.type == 'long':
                current_price = tick.bid
                if position.stop_loss and current_price <= position.stop_loss:
                    self._close_paper_position(position, position.stop_loss)
                elif position.take_profit and current_price >= position.take_profit:
                    self._close_paper_position(position, position.take_profit)
            else:
                current_price = tick.ask
                if position.stop_loss and current_price >= position.stop_loss:
                    self._close_paper_position(position, position.stop_loss)
                elif position.take_profit and current_price <= position.take_profit:
                    self._close_paper_position(position, position.take_profit)

    def _calculate_paper_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL for paper positions."""
        tick = self.broker.get_current_tick(self.symbol)
        if tick is None:
            return 0.0

        symbol_info = self.broker.get_symbol_info(self.symbol)
        contract_size = symbol_info.trade_contract_size if symbol_info else 100000

        unrealized = 0.0
        for position in self._paper_positions:
            if position.type == 'long':
                unrealized += (tick.bid - position.entry_price) * position.size * contract_size
            else:
                unrealized += (position.entry_price - tick.ask) * position.size * contract_size

        return unrealized

    # -------------------------------------------------------------------------
    # Broker sync methods
    # -------------------------------------------------------------------------

    def _sync_with_broker(self):
        """Sync state with broker account."""
        if not self.broker.is_connected:
            return

        account = self.broker.get_account_info()
        if account is None:
            return

        if self.paper_mode:
            self._update_paper_positions()
            self.state.balance = self._paper_balance
            self.state.equity = self._paper_balance + self._calculate_paper_unrealized_pnl()
        else:
            self.state.balance = account.balance
            self.state.equity = account.equity
            self.state.real_balance = account.balance
            self.state.real_equity = account.equity
            self.state.real_margin = account.margin
            self.state.real_free_margin = account.free_margin
            self.state.real_margin_level = account.margin_level

        # Update session PnL
        self.state.session_pnl = self.state.balance - self.state.session_start_balance

        # Update peak equity and drawdown
        self._update_drawdown()

    def _fetch_initial_data(self):
        """Fetch initial market data for observation buffer."""
        if self.data_pipeline is None:
            return

        market_data = self.data_pipeline.fetch_historical_data(
            symbol=self.symbol,
            timeframe='1h',
            n_bars=self.window_size + 50
        )

        if market_data is not None:
            for i in range(len(market_data.close)):
                ohlcv = np.array([
                    market_data.open[i],
                    market_data.high[i],
                    market_data.low[i],
                    market_data.close[i],
                    market_data.volume[i]
                ])
                self._price_buffer.append(ohlcv)

                if market_data.features is not None:
                    self._feature_buffer.append(market_data.features[i])

    def _update_data_buffer(self):
        """Update data buffer with latest tick."""
        tick = self.broker.get_current_tick(self.symbol)
        if tick is None:
            return

        ohlcv = np.array([
            tick.bid,  # Open
            tick.ask,  # High
            tick.bid,  # Low
            tick.bid,  # Close
            0.0  # Volume
        ])

        self._price_buffer.append(ohlcv)

        if len(self._price_buffer) > self._max_buffer_size:
            self._price_buffer = self._price_buffer[-self._max_buffer_size:]
            if self._feature_buffer:
                self._feature_buffer = self._feature_buffer[-self._max_buffer_size:]

    # -------------------------------------------------------------------------
    # Observation and info
    # -------------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        market_obs = self._get_market_observation()

        # Extended account state for live trading
        positions = self._get_open_positions()
        unrealized_pnl = self._get_unrealized_pnl(0.0)

        account_obs = np.array([
            self.state.balance / self.initial_balance,
            self.state.equity / self.initial_balance,
            len(positions),
            1.0 if self._has_position('long') else 0.0,
            1.0 if self._has_position('short') else 0.0,
            unrealized_pnl / self.initial_balance,
            self.state.max_drawdown,
            self.state.total_pnl / self.initial_balance,
            1.0 if self.state.open_status else 0.0,
            1.0 if self.state.close_only else 0.0,
            self.state.session_pnl / self.initial_balance,
            self.state.session_trades / 100.0
        ])

        obs = np.concatenate([market_obs, account_obs]).astype(np.float32)

        # Adjust to match expected dimension
        expected_dim = self.observation_space.shape[0]
        if len(obs) < expected_dim:
            obs = np.pad(obs, (0, expected_dim - len(obs)), mode='constant')
        elif len(obs) > expected_dim:
            obs = obs[:expected_dim]

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get current info dictionary."""
        info = self._get_base_info()

        # Add live-specific info
        info.update({
            'open_status': self.state.open_status,
            'close_only': self.state.close_only,
            'session_pnl': self.state.session_pnl,
            'session_trades': self.state.session_trades,
            'paper_mode': self.paper_mode
        })

        return info

    # -------------------------------------------------------------------------
    # Position callbacks
    # -------------------------------------------------------------------------

    def _on_position_closed(self, change: PositionChange):
        """Callback when position is closed."""
        if change.position:
            pnl = change.details.get('profit', 0.0)
            logger.info(f"Position closed: {change.ticket}, PnL: {pnl:.2f}")

    def _on_sl_hit(self, change: PositionChange):
        """Callback when SL is hit."""
        if change.position:
            logger.info(f"SL hit: {change.ticket}")

    def _on_tp_hit(self, change: PositionChange):
        """Callback when TP is hit."""
        if change.position:
            logger.info(f"TP hit: {change.ticket}")

    # -------------------------------------------------------------------------
    # Rendering and statistics
    # -------------------------------------------------------------------------

    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            info = self._get_info()
            print(
                f"Step: {info['current_step']}, "
                f"Balance: ${info['balance']:.2f}, "
                f"Equity: ${info['equity']:.2f}, "
                f"Positions: {info['positions']}, "
                f"Open: {info['open_status']}"
            )
        elif self.render_mode == 'log':
            info = self._get_info()
            logger.info(
                f"Step {info['current_step']}: "
                f"Balance=${info['balance']:.2f}, Equity=${info['equity']:.2f}"
            )

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get comprehensive episode statistics."""
        stats = super().get_episode_stats()

        if 'error' not in stats:
            stats.update({
                'session_trades': self.state.session_trades,
                'session_pnl': self.state.session_pnl,
                'open_status': self.state.open_status,
                'paper_mode': self.paper_mode
            })

        return stats

    def close(self):
        """Clean up environment."""
        if not self.paper_mode:
            # Close all positions on shutdown (optional)
            pass
        logger.info("Live trading environment closed")
