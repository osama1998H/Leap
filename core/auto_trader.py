"""
Leap Trading System - Auto-Trader Core
Main orchestrator for autonomous trading operations.
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import numpy as np

from core.mt5_broker import MT5BrokerGateway, OrderType
from core.order_manager import OrderManager, TradingSignal, SignalType, OrderExecution
from core.position_sync import PositionSynchronizer, PositionEvent, PositionChange
from core.live_trading_env import LiveTradingEnvironment
from core.data_pipeline import DataPipeline
from core.trading_env import Action
from config.settings import AutoTraderConfig  # Single source of truth for config

if TYPE_CHECKING:
    from core.risk_manager import RiskManager
    from models.transformer import TransformerPredictor
    from models.ppo_agent import PPOAgent
    from training.online_learning import OnlineLearningManager

logger = logging.getLogger(__name__)


class TraderState(Enum):
    """Auto-trader operational states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class TradingSession:
    """Trading session statistics."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    start_balance: float = 0.0
    end_balance: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    signals_generated: int = 0
    signals_executed: int = 0
    signals_rejected: int = 0
    online_adaptations: int = 0

    @property
    def duration(self) -> timedelta:
        end = self.end_time or datetime.now()
        return end - self.start_time

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def pnl_percent(self) -> float:
        if self.start_balance == 0:
            return 0.0
        return (self.end_balance - self.start_balance) / self.start_balance * 100


# Note: AutoTraderConfig is imported from config.settings to maintain single source of truth


class AutoTrader:
    """
    Autonomous trading system orchestrator.

    Integrates:
    - Transformer price prediction
    - PPO reinforcement learning agent
    - MT5 broker for order execution
    - Risk management
    - Online learning for adaptation

    Main Loop:
    1. Fetch latest market data
    2. Generate prediction from Transformer
    3. Get action from PPO agent
    4. Validate with risk manager
    5. Execute via order manager
    6. Update online learning with feedback
    """

    def __init__(
        self,
        broker: MT5BrokerGateway,
        predictor: Optional['TransformerPredictor'] = None,
        agent: Optional['PPOAgent'] = None,
        risk_manager: Optional['RiskManager'] = None,
        online_manager: Optional['OnlineLearningManager'] = None,
        data_pipeline: Optional[DataPipeline] = None,
        config: Optional[AutoTraderConfig] = None
    ):
        """
        Initialize auto-trader.

        Args:
            broker: MT5 broker gateway
            predictor: Transformer predictor model
            agent: PPO agent for decisions
            risk_manager: Risk management system
            online_manager: Online learning manager
            data_pipeline: Data pipeline for features
            config: Auto-trader configuration
        """
        self.broker = broker
        self.predictor = predictor
        self.agent = agent
        self.risk_manager = risk_manager
        self.online_manager = online_manager
        self.data_pipeline = data_pipeline or DataPipeline()
        self.config = config or AutoTraderConfig()

        # Components
        self.order_manager = OrderManager(
            broker=broker,
            risk_manager=risk_manager,
            default_sl_pips=self.config.default_sl_pips,
            default_tp_pips=self.config.default_tp_pips,
            default_risk_percent=self.config.risk_per_trade
        )

        self.position_sync = PositionSynchronizer(
            broker=broker,
            magic_number=broker.magic_number
        )

        # Live environment (one per symbol)
        self.live_envs: Dict[str, LiveTradingEnvironment] = {}

        # State
        self.state = TraderState.STOPPED
        self.session: Optional[TradingSession] = None
        self._stop_event = threading.Event()
        self._trading_thread: Optional[threading.Thread] = None

        # Trading state
        self._last_bar_time: Dict[str, datetime] = {}
        self._daily_start_balance: float = 0.0
        self._last_heartbeat: Optional[datetime] = None
        self._adaptations_today: int = 0
        self._last_adaptation_date: Optional[datetime] = None
        self._daily_loss_limit_hit: bool = False  # Hysteresis flag for daily loss warning
        self._consecutive_errors: int = 0  # Track consecutive loop errors
        self._max_consecutive_errors: int = 10  # Threshold before ERROR state

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_trade': [],
            'on_signal': [],
            'on_error': [],
            'on_adaptation': [],
            'on_state_change': []
        }

        # Trade history for online learning
        self._recent_trades: List[OrderExecution] = []
        self._max_trade_history = 1000
        # Store signal data for online learning (keyed by ticket)
        self._pending_learning_data: Dict[int, Dict[str, Any]] = {}

        # Register position callbacks
        self.position_sync.register_callback(
            PositionEvent.CLOSED, self._on_position_closed
        )
        self.position_sync.register_callback(
            PositionEvent.SL_HIT, self._on_position_closed
        )
        self.position_sync.register_callback(
            PositionEvent.TP_HIT, self._on_position_closed
        )

        logger.info("AutoTrader initialized")

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def start(self) -> bool:
        """
        Start the auto-trader.

        Returns:
            True if started successfully
        """
        if self.state == TraderState.RUNNING:
            logger.warning("AutoTrader already running")
            return False

        logger.info("Starting AutoTrader...")
        self._set_state(TraderState.STARTING)

        # Connect to broker
        if not self.broker.is_connected:
            if not self.broker.connect():
                logger.error("Failed to connect to broker")
                self._set_state(TraderState.ERROR)
                return False

        # Initialize session
        account = self.broker.get_account_info()
        self.session = TradingSession(
            start_balance=account.balance if account else 0.0
        )
        self._daily_start_balance = self.session.start_balance

        # Initialize live environments
        try:
            self._initialize_environments()
        except Exception:
            logger.exception("Failed to initialize trading environments")
            self._set_state(TraderState.ERROR)
            return False

        if not self.live_envs:
            logger.error("No trading environments initialized")
            self._set_state(TraderState.ERROR)
            return False

        # Reset stop event
        self._stop_event.clear()

        # Start trading thread
        self._trading_thread = threading.Thread(
            target=self._trading_loop,
            daemon=True,
            name="AutoTrader"
        )
        self._trading_thread.start()

        self._set_state(TraderState.RUNNING)
        logger.info(f"AutoTrader started (paper_mode={self.config.paper_mode})")

        return True

    def stop(self, wait: bool = True, timeout: float = 30.0):
        """
        Stop the auto-trader.

        Args:
            wait: Wait for thread to finish
            timeout: Maximum wait time in seconds
        """
        if self.state in [TraderState.STOPPED, TraderState.STOPPING]:
            return

        logger.info("Stopping AutoTrader...")
        self._set_state(TraderState.STOPPING)

        # Signal stop
        self._stop_event.set()

        # Wait for thread
        if wait and self._trading_thread and self._trading_thread.is_alive():
            self._trading_thread.join(timeout=timeout)

        # Finalize session
        if self.session:
            account = self.broker.get_account_info()
            self.session.end_time = datetime.now()
            self.session.end_balance = account.balance if account else self.session.start_balance

        self._set_state(TraderState.STOPPED)
        logger.info("AutoTrader stopped")

    def pause(self):
        """Pause trading (keep monitoring but don't trade)."""
        if self.state == TraderState.RUNNING:
            self._set_state(TraderState.PAUSED)
            logger.info("AutoTrader paused")

    def resume(self):
        """Resume trading after pause."""
        if self.state == TraderState.PAUSED:
            self._set_state(TraderState.RUNNING)
            logger.info("AutoTrader resumed")

    def set_open_status(self, symbol: str, can_open: bool):
        """Set open status for a symbol."""
        if symbol in self.live_envs:
            self.live_envs[symbol].set_open_status(can_open)

    def set_close_only(self, symbol: str, close_only: bool):
        """Set close-only mode for a symbol."""
        if symbol in self.live_envs:
            self.live_envs[symbol].set_close_only(close_only)

    def close_all_positions(self, symbol: Optional[str] = None):
        """Close all positions."""
        self.order_manager.close_all_positions(symbol)

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        account = self.broker.get_account_info()

        status = {
            'state': self.state.value,
            'paper_mode': self.config.paper_mode,
            'symbols': self.config.symbols,
            'connected': self.broker.is_connected,
            'balance': account.balance if account else 0.0,
            'equity': account.equity if account else 0.0,
            'positions': self.position_sync.get_positions_count(),
            'session': None
        }

        if self.session:
            status['session'] = {
                'duration': str(self.session.duration),
                'total_trades': self.session.total_trades,
                'win_rate': self.session.win_rate,
                'pnl': self.session.total_pnl,
                'signals_generated': self.session.signals_generated,
                'signals_executed': self.session.signals_executed
            }

        return status

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        stats = {
            'trader_state': self.state.value,
            'session': None,
            'order_manager': self.order_manager.get_statistics(),
            'position_sync': self.position_sync.get_statistics(),
            'config': {
                'symbols': self.config.symbols,
                'risk_per_trade': self.config.risk_per_trade,
                'max_positions': self.config.max_positions,
                'paper_mode': self.config.paper_mode
            }
        }

        if self.session:
            stats['session'] = {
                'start_time': str(self.session.start_time),
                'duration': str(self.session.duration),
                'start_balance': self.session.start_balance,
                'current_balance': self.session.end_balance or self.session.start_balance,
                'pnl': self.session.total_pnl,
                'pnl_percent': self.session.pnl_percent,
                'total_trades': self.session.total_trades,
                'winning_trades': self.session.winning_trades,
                'losing_trades': self.session.losing_trades,
                'win_rate': self.session.win_rate,
                'max_drawdown': self.session.max_drawdown,
                'signals_generated': self.session.signals_generated,
                'signals_executed': self.session.signals_executed,
                'signals_rejected': self.session.signals_rejected,
                'online_adaptations': self.session.online_adaptations
            }

        return stats

    def _initialize_environments(self):
        """Initialize live trading environments for each symbol."""
        for symbol in self.config.symbols:
            env = LiveTradingEnvironment(
                broker=self.broker,
                symbol=symbol,
                data_pipeline=self.data_pipeline,
                risk_manager=self.risk_manager,
                max_positions=self.config.max_positions,
                default_sl_pips=self.config.default_sl_pips,
                default_tp_pips=self.config.default_tp_pips,
                risk_per_trade=self.config.risk_per_trade,
                paper_mode=self.config.paper_mode
            )
            env.reset()
            self.live_envs[symbol] = env
            logger.info(f"Initialized environment for {symbol}")

    def _trading_loop(self):
        """Main trading loop."""
        logger.info("Trading loop started")

        while not self._stop_event.is_set():
            try:
                # Check if we're in ERROR state (should exit loop)
                if self.state == TraderState.ERROR:
                    logger.warning("Trading loop exiting due to ERROR state")
                    break

                # Check if we should trade
                if self.state != TraderState.RUNNING:
                    time.sleep(self.config.loop_interval)
                    continue

                if not self._is_trading_time():
                    time.sleep(self.config.loop_interval)
                    continue

                # Heartbeat
                self._heartbeat()

                # Check daily limits
                if self._check_daily_limits():
                    time.sleep(self.config.loop_interval)
                    continue

                # Sync positions once per loop (not per symbol)
                self.position_sync.sync()

                # Run trading cycle for each symbol
                for symbol in self.config.symbols:
                    self._run_trading_cycle(symbol)

                # Check for online learning adaptation
                self._check_adaptation()

                # Reset consecutive error counter on successful iteration
                self._consecutive_errors = 0

                # Sleep until next cycle
                time.sleep(self.config.loop_interval)

            except Exception as e:
                self._consecutive_errors += 1
                logger.error(
                    f"Error in trading loop ({self._consecutive_errors}/{self._max_consecutive_errors}): {e}",
                    exc_info=True
                )
                self._fire_callback('on_error', {'error': str(e), 'consecutive_count': self._consecutive_errors})

                # Transition to ERROR state after too many consecutive failures
                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.error(
                        f"Max consecutive errors ({self._max_consecutive_errors}) reached. "
                        "Transitioning to ERROR state."
                    )
                    self._set_state(TraderState.ERROR)
                    break

                time.sleep(5.0)  # Longer sleep on error

        logger.info("Trading loop stopped")

    def _run_trading_cycle(self, symbol: str):
        """Run one trading cycle for a symbol."""
        # Note: position_sync.sync() is called once in the main loop, not per symbol

        # Check if new bar
        if not self._is_new_bar(symbol):
            return

        # Get environment
        env = self.live_envs.get(symbol)
        if env is None:
            return

        # Generate trading signal
        signal = self._generate_signal(symbol)

        if self.session:
            self.session.signals_generated += 1

        self._fire_callback('on_signal', {'symbol': symbol, 'signal': signal})

        # Execute signal
        if signal.signal_type != SignalType.HOLD:
            execution = self.order_manager.execute_signal(signal)

            if execution.executed:
                if self.session:
                    self.session.signals_executed += 1
                    self.session.total_trades += 1

                self._recent_trades.append(execution)
                self._fire_callback('on_trade', {'execution': execution})

                # Store learning data for when position closes
                if execution.ticket and self.online_manager is not None:
                    self._pending_learning_data[execution.ticket] = {
                        'observation': env._get_observation() if env else None,
                        'predicted_return': signal.metadata.get('predicted_return', 0.0),
                        'action': signal.signal_type.value,
                        'entry_price': execution.entry_price,
                        'timestamp': datetime.now(timezone.utc)
                    }

                if self.config.log_trades:
                    logger.info(
                        f"Trade executed: {signal.signal_type.value} {symbol} "
                        f"@ {execution.entry_price} (ticket: {execution.ticket})"
                    )
            else:
                if self.session:
                    self.session.signals_rejected += 1

    def _generate_signal(self, symbol: str) -> TradingSignal:
        """Generate trading signal using models."""
        env = self.live_envs.get(symbol)
        if env is None:
            return TradingSignal(signal_type=SignalType.HOLD, symbol=symbol)

        # Get observation from environment
        obs = env._get_observation()

        # Get prediction from Transformer (if available)
        predicted_return = 0.0
        prediction_confidence = 0.5

        if self.predictor is not None:
            try:
                # Reshape observation for predictor
                # This assumes the predictor can handle the observation format
                X = obs.reshape(1, -1)  # Adjust shape as needed

                prediction = self.predictor.predict(X)
                predicted_return = prediction.get('prediction', np.array([[0.0]]))[0, 0]
                # Clamp confidence to [0, 1] range
                uncertainty = prediction.get('uncertainty', 0.5)
                prediction_confidence = max(0.0, min(1.0, 1.0 - uncertainty))
            except Exception:
                logger.exception("Prediction failed")

        # Get action from PPO agent (if available)
        agent_action = Action.HOLD

        if self.agent is not None:
            try:
                # select_action returns (action, log_prob, value) - we only need action
                action, _, _ = self.agent.select_action(obs, deterministic=True)
                agent_action = Action(action)
            except Exception:
                logger.exception("Agent action failed")

        # Combine prediction and agent decision
        signal_type = self._combine_signals(
            predicted_return=predicted_return,
            agent_action=agent_action,
            confidence=prediction_confidence,
            symbol=symbol
        )

        return TradingSignal(
            signal_type=signal_type,
            symbol=symbol,
            confidence=prediction_confidence,
            stop_loss_pips=self.config.default_sl_pips,
            take_profit_pips=self.config.default_tp_pips,
            risk_percent=self.config.risk_per_trade,
            source="auto_trader",
            metadata={
                'predicted_return': predicted_return,
                'agent_action': agent_action.name
            }
        )

    def _combine_signals(
        self,
        predicted_return: float,
        agent_action: Action,
        confidence: float,
        symbol: str
    ) -> SignalType:
        """Combine prediction and agent action into final signal."""
        # Check confidence threshold
        if confidence < self.config.min_confidence:
            return SignalType.HOLD

        # Check if environment allows opening
        env = self.live_envs.get(symbol)
        if env and not env.open_status:
            if agent_action == Action.CLOSE:
                return SignalType.CLOSE
            return SignalType.HOLD

        # Strategy: Use both prediction and agent
        # Priority: Agent decision, validated by prediction direction

        if agent_action == Action.CLOSE:
            return SignalType.CLOSE

        if agent_action == Action.BUY:
            # Validate with prediction
            if predicted_return >= self.config.prediction_threshold:
                return SignalType.BUY
            elif predicted_return < -self.config.prediction_threshold:
                # Contradiction - hold
                return SignalType.HOLD
            else:
                # Weak prediction - trust agent
                return SignalType.BUY

        if agent_action == Action.SELL:
            # Validate with prediction
            if predicted_return <= -self.config.prediction_threshold:
                return SignalType.SELL
            elif predicted_return > self.config.prediction_threshold:
                # Contradiction - hold
                return SignalType.HOLD
            else:
                # Weak prediction - trust agent
                return SignalType.SELL

        # Default: HOLD
        return SignalType.HOLD

    def _is_new_bar(self, symbol: str) -> bool:
        """Check if a new bar has formed (using UTC for consistency)."""
        now = datetime.now(timezone.utc)

        # Simple time-based check
        last_bar = self._last_bar_time.get(symbol)
        if last_bar is None:
            self._last_bar_time[symbol] = now
            return True

        elapsed = (now - last_bar).total_seconds()
        if elapsed >= self.config.bar_interval:
            self._last_bar_time[symbol] = now
            return True

        return False

    def _is_trading_time(self) -> bool:
        """Check if we're within trading hours (UTC)."""
        # Use UTC time as documented in config
        now = datetime.now(timezone.utc)

        # Check day (0=Monday, 6=Sunday)
        if now.weekday() not in self.config.trading_days:
            return False

        # Check hour (in UTC)
        if self.config.trading_start_hour == 0 and self.config.trading_end_hour == 24:
            return True

        hour = now.hour
        if self.config.trading_start_hour < self.config.trading_end_hour:
            return self.config.trading_start_hour <= hour < self.config.trading_end_hour
        else:
            # Crosses midnight
            return hour >= self.config.trading_start_hour or hour < self.config.trading_end_hour

    def _check_daily_limits(self) -> bool:
        """Check if daily limits are hit. Returns True if trading should stop."""
        # Use UTC for consistency with trading hours
        now = datetime.now(timezone.utc)

        # Reset daily counters at day change (UTC)
        if self._last_adaptation_date and self._last_adaptation_date.date() != now.date():
            self._adaptations_today = 0
            self._daily_loss_limit_hit = False  # Reset hysteresis flag

            account = self.broker.get_account_info()
            if account:
                self._daily_start_balance = account.balance

        self._last_adaptation_date = now

        # Check daily loss limit with hysteresis (only log once)
        account = self.broker.get_account_info()
        if account and self._daily_start_balance > 0:
            daily_pnl = (account.balance - self._daily_start_balance) / self._daily_start_balance

            if daily_pnl <= -self.config.max_daily_loss:
                if not getattr(self, '_daily_loss_limit_hit', False):
                    logger.warning(f"Daily loss limit hit: {daily_pnl:.2%}. Trading paused until next day.")
                    self._daily_loss_limit_hit = True
                return True

        return False

    def _check_adaptation(self):
        """Check if online learning adaptation should be triggered."""
        if not self.config.enable_online_learning:
            return

        if self.online_manager is None:
            return

        # Check adaptation frequency
        if len(self._recent_trades) < self.config.adaptation_frequency:
            return

        # Check daily limit
        if self._adaptations_today >= self.config.max_adaptations_per_day:
            return

        # Trigger adaptation
        try:
            self._trigger_adaptation()
            self._adaptations_today += 1

            if self.session:
                self.session.online_adaptations += 1

            self._fire_callback('on_adaptation', {
                'count': self._adaptations_today,
                'trades_since_last': len(self._recent_trades)
            })

            # Clear recent trades
            self._recent_trades.clear()

        except Exception:
            logger.exception("Adaptation check failed")

    def _trigger_adaptation(self):
        """Trigger online learning adaptation.

        Note: Learning data is accumulated via step() calls in _on_position_closed.
        This method triggers the actual model adaptation when enough data is collected.
        """
        if self.online_manager is None:
            logger.warning("Online manager not available, skipping adaptation")
            return

        # Count trades that will be used for adaptation context
        trade_count = sum(1 for ex in self._recent_trades if ex.executed)
        logger.info(f"Triggering online learning adaptation with {trade_count} recent trades...")

        if trade_count == 0:
            logger.info("No executed trades for adaptation context, skipping")
            return

        # Perform adaptation using accumulated data from step() calls
        try:
            adaptation_result = self.online_manager._perform_adaptation()

            # Log detailed results
            if adaptation_result:
                predictor_updated = adaptation_result.get('predictor_updated', False)
                agent_updated = adaptation_result.get('agent_updated', False)
                regime = adaptation_result.get('regime', 'unknown')

                logger.info(
                    f"Adaptation completed: predictor_updated={predictor_updated}, "
                    f"agent_updated={agent_updated}, regime={regime}"
                )

                if adaptation_result.get('predictor_loss') is not None:
                    logger.info(f"  Predictor loss: {adaptation_result['predictor_loss']:.6f}")
                if adaptation_result.get('agent_loss') is not None:
                    logger.info(f"  Agent loss: {adaptation_result['agent_loss']:.6f}")
            else:
                logger.info("Adaptation completed with no updates (insufficient data in buffers)")

        except Exception:
            logger.exception("Adaptation failed")

    def _on_position_closed(self, change: PositionChange):
        """Handle position closed event."""
        if change.position is None:
            return

        profit = change.details.get('profit', 0.0)
        ticket = change.ticket

        if self.session:
            self.session.total_pnl += profit
            if profit > 0:
                self.session.winning_trades += 1
            else:
                self.session.losing_trades += 1

            # Update drawdown
            account = self.broker.get_account_info()
            if account and self.session.start_balance > 0:
                drawdown = (self.session.start_balance - account.balance) / self.session.start_balance
                if drawdown > self.session.max_drawdown:
                    self.session.max_drawdown = drawdown

        # Feed data to online learning manager
        if self.online_manager is not None and ticket in self._pending_learning_data:
            learning_data = self._pending_learning_data.pop(ticket)
            try:
                # Calculate actual return (profit as percentage of entry)
                entry_price = learning_data.get('entry_price', 1.0)
                actual_return = profit / entry_price if entry_price else 0.0

                # Map action string to int for step()
                action_map = {'buy': 1, 'sell': 2, 'close': 3, 'hold': 0}
                action_int = action_map.get(learning_data.get('action', 'hold'), 0)

                # Prepare market data for step()
                market_data = {
                    'state': learning_data.get('observation'),
                    'features': learning_data.get('observation'),
                    'close': entry_price,
                    'returns': actual_return
                }

                # Call step() to accumulate learning data
                step_result = self.online_manager.step(
                    market_data=market_data,
                    prediction=learning_data.get('predicted_return', 0.0),
                    actual=actual_return,
                    trading_action=action_int,
                    trading_reward=profit,
                    done=True
                )

                logger.debug(f"Online learning step completed for ticket {ticket}: {step_result}")

            except Exception:
                logger.exception(f"Failed to feed learning data for ticket {ticket}")

    def _heartbeat(self):
        """Log heartbeat status (using UTC for consistency)."""
        now = datetime.now(timezone.utc)

        if self._last_heartbeat is None:
            self._last_heartbeat = now

        elapsed = (now - self._last_heartbeat).total_seconds()
        if elapsed >= self.config.heartbeat_interval:
            self._last_heartbeat = now

            try:
                account = self.broker.get_account_info()
                positions = self.position_sync.get_positions_count()

                if account:
                    logger.info(
                        f"Heartbeat: balance=${account.balance:.2f}, "
                        f"equity=${account.equity:.2f}, "
                        f"positions={positions}"
                    )
                else:
                    logger.info("Heartbeat: disconnected")
            except Exception:
                logger.exception("Heartbeat failed to get status")

    def _set_state(self, new_state: TraderState):
        """Set trader state and fire callback."""
        old_state = self.state
        self.state = new_state
        self._fire_callback('on_state_change', {
            'old_state': old_state.value,
            'new_state': new_state.value
        })

    def _fire_callback(self, event: str, data: Dict[str, Any]):
        """Fire registered callbacks for an event."""
        callbacks = self._callbacks.get(event, [])
        for callback in callbacks:
            try:
                callback(data)
            except Exception:
                logger.exception(f"Callback error for {event}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
