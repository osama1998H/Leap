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
from core.mt5_broker import MT5BrokerGateway, MT5Position
from core.position_sync import PositionSynchronizer, PositionEvent, PositionChange
from core.order_manager import OrderManager, TradingSignal, SignalType
from utils.pnl_calculator import calculate_pnl, calculate_unrealized_pnl
from utils.position_sizing import calculate_risk_based_size

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

    # Constants for observation space dimensions
    N_PRICE_FEATURES = 5  # OHLCV
    N_BASE_ACCOUNT_FEATURES = 8  # From base class
    N_LIVE_EXTRA_FEATURES = 4  # Live-specific: open_status, close_only, session_pnl, session_trades
    DEFAULT_ADDITIONAL_FEATURES = 100  # Default when feature_dim not provided

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
        paper_mode: bool = True,
        feature_dim: Optional[int] = None,
        match_training_obs: bool = False
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
            feature_dim: Number of additional features (from FeatureEngineer).
                         If None, attempts to infer from data_pipeline or uses default.
            match_training_obs: If True, return observations with 8 account features
                         (matching training environment) instead of 12. This ensures
                         compatibility with models trained on TradingEnvironment.
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
        self.match_training_obs = match_training_obs

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

        # Determine feature dimension (MAJOR-1 fix: dynamic instead of hardcoded)
        self.n_additional_features = self._resolve_feature_dim(feature_dim, data_pipeline)

        # Calculate observation dimension
        # When match_training_obs=True, use 8 account features (matching training env)
        # Otherwise use 12 (8 base + 4 live-specific)
        if self.match_training_obs:
            self.n_account_features = self.N_BASE_ACCOUNT_FEATURES  # 8
        else:
            self.n_account_features = self.N_BASE_ACCOUNT_FEATURES + self.N_LIVE_EXTRA_FEATURES  # 12

        obs_dim = (
            window_size * (self.N_PRICE_FEATURES + self.n_additional_features) +
            self.n_account_features
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
    # Helper methods
    # -------------------------------------------------------------------------

    def _resolve_feature_dim(
        self,
        feature_dim: Optional[int],
        data_pipeline
    ) -> int:
        """
        Resolve the number of additional features for observation space.

        Priority:
        1. Explicit feature_dim parameter
        2. Infer from data_pipeline.feature_count
        3. Fall back to DEFAULT_ADDITIONAL_FEATURES

        Args:
            feature_dim: Explicit feature dimension (if provided)
            data_pipeline: Data pipeline that may have feature_count attribute

        Returns:
            Number of additional features to use
        """
        if feature_dim is not None:
            logger.debug(f"Using explicit feature_dim: {feature_dim}")
            return feature_dim

        if data_pipeline is not None and hasattr(data_pipeline, 'feature_count'):
            inferred = data_pipeline.feature_count
            logger.debug(f"Inferred feature_dim from data_pipeline: {inferred}")
            return inferred

        logger.debug(f"Using default feature_dim: {self.DEFAULT_ADDITIONAL_FEATURES}")
        return self.DEFAULT_ADDITIONAL_FEATURES

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
                price_window = np.pad(prices, (padding_size * self.N_PRICE_FEATURES, 0), mode='edge')
            else:
                price_window = np.zeros(self.window_size * self.N_PRICE_FEATURES)

        # Feature window (MAJOR-1 fix: use dynamic n_additional_features)
        if self._feature_buffer and len(self._feature_buffer) >= self.window_size:
            feature_window = np.array(self._feature_buffer[-self.window_size:]).flatten()
        else:
            feature_window = np.zeros(self.window_size * self.n_additional_features)

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

        # Calculate SL/TP prices for risk validation
        symbol_info = self.broker.get_symbol_info(self.symbol)
        pip_size = symbol_info.point * 10 if symbol_info else 0.0001

        if direction == 'long':
            entry_price = price
            stop_loss_price = entry_price - (self.default_sl_pips * pip_size)
            take_profit_price = entry_price + (self.default_tp_pips * pip_size)
        else:
            entry_price = price
            stop_loss_price = entry_price + (self.default_sl_pips * pip_size)
            take_profit_price = entry_price - (self.default_tp_pips * pip_size)

        # Validate trade with RiskManager before opening (should_take_trade pattern)
        if self.risk_manager is not None:
            should_trade, reason = self.risk_manager.should_take_trade(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                direction=direction
            )
            if not should_trade:
                logger.info(f"Trade rejected by risk manager: {reason}")
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
        """Get list of open positions from single source of truth.

        Returns self.state.positions, which is kept in sync by _sync_positions_to_state().
        """
        # Ensure positions are synced (state.positions is the single source of truth)
        # Note: _sync_with_broker() already calls _sync_positions_to_state(),
        # but we sync here too for direct calls to _get_open_positions()
        self._sync_positions_to_state()
        return self.state.positions

    def _has_position(self, direction: str) -> bool:
        """Check if we have a position in given direction.

        Uses self.state.positions as the single source of truth.
        """
        # Use unified position storage (state.positions)
        positions = self._get_open_positions()
        return any(p.type == direction for p in positions)

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

        # Calculate volume - use RiskManager when available (matches Backtester pattern)
        if self.risk_manager is not None:
            volume = self.risk_manager.calculate_position_size(
                entry_price=entry_price,
                stop_loss_price=sl
            )
            if volume <= 0:
                logger.debug("Position size is zero or negative, skipping paper trade")
                return
        else:
            # Fallback: use centralized position sizing utility with symbol-specific pip value
            try:
                pip_value = symbol_info.trade_tick_value * (pip_size / symbol_info.trade_tick_size)
            except (AttributeError, ZeroDivisionError, TypeError) as e:
                logger.warning(
                    f"Failed to calculate pip value for {self.symbol} from symbol info: {e}. "
                    f"Using fallback value of 10.0 (appropriate for major pairs only)"
                )
                pip_value = 10.0  # Fallback for major pairs

            if pip_value <= 0:
                logger.error(f"Invalid pip_value {pip_value} for {self.symbol}, using fallback 10.0")
                pip_value = 10.0

            volume = calculate_risk_based_size(
                self._paper_balance, self.risk_per_trade, self.default_sl_pips, pip_value
            )

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
            exit_price = tick.bid if position.type == 'long' else tick.ask
            pnl = calculate_pnl(
                position.entry_price, exit_price, position.size, position.type, contract_size
            )

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

        pnl = calculate_pnl(
            position.entry_price, exit_price, position.size, position.type, contract_size
        )

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

        return sum(
            calculate_unrealized_pnl(
                position.entry_price,
                tick.bid if position.type == 'long' else tick.ask,
                position.size,
                position.type,
                contract_size
            )
            for position in self._paper_positions
        )

    # -------------------------------------------------------------------------
    # Broker sync methods
    # -------------------------------------------------------------------------

    def _mt5_position_to_position(self, mt5_pos: MT5Position) -> Position:
        """
        Convert MT5Position to Position dataclass.

        Args:
            mt5_pos: MT5 position object from broker

        Returns:
            Position dataclass instance compatible with TradingState
        """
        # Convert type from int (0=BUY, 1=SELL) to string ('long', 'short')
        direction = 'long' if mt5_pos.type == 0 else 'short'

        return Position(
            type=direction,
            entry_price=mt5_pos.price_open,
            size=mt5_pos.volume,
            entry_time=int(mt5_pos.time.timestamp()) if mt5_pos.time else self.current_step,
            stop_loss=mt5_pos.sl if mt5_pos.sl > 0 else None,
            take_profit=mt5_pos.tp if mt5_pos.tp > 0 else None
        )

    def _sync_positions_to_state(self):
        """
        Sync positions to self.state.positions (single source of truth).

        This ensures self.state.positions always reflects the current positions,
        whether from paper trading or live broker.
        """
        if self.paper_mode:
            # Copy paper positions to state
            self.state.positions = list(self._paper_positions)
        else:
            # Get positions from broker via position_sync and convert to Position objects
            broker_positions = self.position_sync.get_positions(self.symbol)
            # Convert MT5Position objects to Position dataclass instances
            self.state.positions = [self._mt5_position_to_position(p) for p in broker_positions]

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

        # Sync positions to state.positions (single source of truth)
        self._sync_positions_to_state()

        # Update session PnL
        self.state.session_pnl = self.state.balance - self.state.session_start_balance

        # Update peak equity and drawdown
        self._update_drawdown()

    def _fetch_initial_data(self):
        """Fetch initial market data for observation buffer.

        Performance: Uses vectorized stacking instead of per-element loops.
        """
        if self.data_pipeline is None:
            return

        market_data = self.data_pipeline.fetch_historical_data(
            symbol=self.symbol,
            timeframe='1h',
            n_bars=self.window_size + 50
        )

        if market_data is not None:
            # Vectorized: Stack all OHLCV data at once instead of looping
            ohlcv_array = np.column_stack([
                market_data.open,
                market_data.high,
                market_data.low,
                market_data.close,
                market_data.volume
            ])

            # Extend buffer with all rows at once (convert to list of arrays)
            self._price_buffer.extend([ohlcv_array[i] for i in range(len(ohlcv_array))])

            # Features - also vectorized
            if market_data.features is not None:
                self._feature_buffer.extend([market_data.features[i] for i in range(len(market_data.features))])

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

    def _get_account_observation(self) -> np.ndarray:
        """
        Get extended account observation for live trading.

        MAJOR-2 fix: Extends base class account observation (8 features)
        with 4 live-specific features for a total of 12 features.

        Base class features (8):
            balance_norm, equity_norm, n_positions, has_long, has_short,
            unrealized_pnl_norm, max_drawdown, total_pnl_norm

        Live-specific features (4):
            open_status, close_only, session_pnl_norm, session_trades_norm
        """
        # Get base class account observation (8 features)
        base_account_obs = super()._get_account_observation()

        # Add live trading specific features (4 features)
        live_extras = np.array([
            1.0 if self.state.open_status else 0.0,
            1.0 if self.state.close_only else 0.0,
            self.state.session_pnl / self.initial_balance,
            self.state.session_trades / 100.0
        ], dtype=np.float32)

        # Concatenate base + live features (8 + 4 = 12)
        return np.concatenate([base_account_obs, live_extras])

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        market_obs = self._get_market_observation()

        # Get account observation based on match_training_obs setting
        if self.match_training_obs:
            # Use only base class account observation (8 features) to match training env
            account_obs = super()._get_account_observation()
        else:
            # Use extended account observation (8 base + 4 live = 12 features)
            account_obs = self._get_account_observation()

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
