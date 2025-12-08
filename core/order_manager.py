"""
Leap Trading System - Order Manager
Handles order execution, validation, and position sizing for the auto-trader.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from core.mt5_broker import MT5BrokerGateway, OrderResult, SymbolInfo, TickInfo
    from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """Represents a trading signal from the model."""
    signal_type: SignalType
    symbol: str
    confidence: float = 1.0
    stop_loss_pips: Optional[float] = None
    take_profit_pips: Optional[float] = None
    risk_percent: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "model"  # 'model', 'manual', 'risk_manager'
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        return self.signal_type in [SignalType.BUY, SignalType.SELL]

    @property
    def is_exit(self) -> bool:
        return self.signal_type in [SignalType.CLOSE, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]

    @property
    def is_hold(self) -> bool:
        return self.signal_type == SignalType.HOLD


@dataclass
class OrderExecution:
    """Record of an order execution."""
    signal: TradingSignal
    executed: bool
    ticket: int = 0
    volume: float = 0.0
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    error_message: str = ""
    execution_time: datetime = field(default_factory=datetime.now)


class OrderManager:
    """
    Manages order execution for the auto-trader.

    Responsibilities:
    - Validate signals before execution
    - Calculate appropriate position sizing
    - Set SL/TP based on strategy parameters
    - Execute orders through the broker gateway
    - Handle execution failures and retries
    - Log all trading activity
    """

    def __init__(
        self,
        broker: 'MT5BrokerGateway',
        risk_manager: Optional['RiskManager'] = None,
        default_sl_pips: float = 50.0,
        default_tp_pips: float = 100.0,
        default_risk_percent: float = 0.01,
        max_spread_pips: float = 5.0,
        min_confidence: float = 0.5
    ):
        """
        Initialize order manager.

        Args:
            broker: MT5 broker gateway
            risk_manager: Risk manager for position sizing
            default_sl_pips: Default stop loss in pips
            default_tp_pips: Default take profit in pips
            default_risk_percent: Default risk per trade (1%)
            max_spread_pips: Maximum allowed spread in pips
            min_confidence: Minimum confidence for execution
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.default_sl_pips = default_sl_pips
        self.default_tp_pips = default_tp_pips
        self.default_risk_percent = default_risk_percent
        self.max_spread_pips = max_spread_pips
        self.min_confidence = min_confidence

        # Execution history
        self.executions: List[OrderExecution] = []

        # Statistics
        self.stats = {
            'total_signals': 0,
            'executed': 0,
            'rejected': 0,
            'failed': 0
        }

    def execute_signal(self, signal: TradingSignal) -> OrderExecution:
        """
        Execute a trading signal.

        Args:
            signal: Trading signal to execute

        Returns:
            OrderExecution record
        """
        self.stats['total_signals'] += 1

        # Handle HOLD signals
        if signal.is_hold:
            execution = OrderExecution(
                signal=signal,
                executed=True,
                error_message="Hold signal - no action"
            )
            self.executions.append(execution)
            return execution

        # Handle exit signals
        if signal.is_exit:
            return self._execute_close_signal(signal)

        # Handle entry signals
        return self._execute_entry_signal(signal)

    def _execute_entry_signal(self, signal: TradingSignal) -> OrderExecution:
        """Execute an entry signal (BUY or SELL)."""
        # Validate signal
        is_valid, reason = self._validate_signal(signal)
        if not is_valid:
            self.stats['rejected'] += 1
            execution = OrderExecution(
                signal=signal,
                executed=False,
                error_message=reason
            )
            self.executions.append(execution)
            logger.info(f"Signal rejected: {reason}")
            return execution

        # Get symbol info
        symbol_info = self.broker.get_symbol_info(signal.symbol)
        if symbol_info is None:
            self.stats['failed'] += 1
            execution = OrderExecution(
                signal=signal,
                executed=False,
                error_message=f"Failed to get symbol info for {signal.symbol}"
            )
            self.executions.append(execution)
            return execution

        # Get current tick
        tick = self.broker.get_current_tick(signal.symbol)
        if tick is None:
            self.stats['failed'] += 1
            execution = OrderExecution(
                signal=signal,
                executed=False,
                error_message=f"Failed to get tick for {signal.symbol}"
            )
            self.executions.append(execution)
            return execution

        # Calculate position size
        volume, sl, tp = self._calculate_position_params(signal, symbol_info, tick)

        if volume <= 0:
            self.stats['rejected'] += 1
            execution = OrderExecution(
                signal=signal,
                executed=False,
                error_message="Calculated volume is zero or negative"
            )
            self.executions.append(execution)
            return execution

        # Execute order
        from core.mt5_broker import OrderType

        order_type = OrderType.BUY if signal.signal_type == SignalType.BUY else OrderType.SELL

        result = self.broker.send_market_order(
            symbol=signal.symbol,
            order_type=order_type,
            volume=volume,
            sl=sl,
            tp=tp,
            comment=f"Leap {signal.source} {signal.confidence:.2f}"
        )

        if result.success:
            self.stats['executed'] += 1

            # Notify risk manager
            if self.risk_manager:
                entry_price = result.price if result.price else (tick.ask if order_type == OrderType.BUY else tick.bid)
                notional = volume * entry_price * (symbol_info.trade_contract_size if symbol_info else 100000)
                self.risk_manager.on_position_opened(notional)

            execution = OrderExecution(
                signal=signal,
                executed=True,
                ticket=result.ticket,
                volume=volume,
                entry_price=result.price,
                sl=sl,
                tp=tp
            )
            logger.info(
                f"Order executed: {signal.signal_type.value} {signal.symbol} "
                f"{volume} lots @ {result.price} (SL: {sl}, TP: {tp})"
            )
        else:
            self.stats['failed'] += 1
            execution = OrderExecution(
                signal=signal,
                executed=False,
                error_message=result.comment
            )
            logger.warning(f"Order failed: {result.comment}")

        self.executions.append(execution)
        return execution

    def _execute_close_signal(self, signal: TradingSignal) -> OrderExecution:
        """Execute a close signal."""
        positions = self.broker.get_positions(symbol=signal.symbol)

        if not positions:
            execution = OrderExecution(
                signal=signal,
                executed=True,
                error_message="No positions to close"
            )
            self.executions.append(execution)
            return execution

        # Filter positions based on signal type
        positions_to_close = []
        if signal.signal_type == SignalType.CLOSE:
            positions_to_close = positions
        elif signal.signal_type == SignalType.CLOSE_LONG:
            positions_to_close = [p for p in positions if p.is_long]
        elif signal.signal_type == SignalType.CLOSE_SHORT:
            positions_to_close = [p for p in positions if p.is_short]

        if not positions_to_close:
            execution = OrderExecution(
                signal=signal,
                executed=True,
                error_message="No matching positions to close"
            )
            self.executions.append(execution)
            return execution

        # Close positions
        all_success = True
        closed_tickets = []
        total_volume = 0.0

        for position in positions_to_close:
            result = self.broker.close_position(
                ticket=position.ticket,
                comment=f"Leap {signal.source} close"
            )

            if result.success:
                closed_tickets.append(position.ticket)
                total_volume += position.volume

                # Notify risk manager
                if self.risk_manager:
                    notional = position.volume * position.price_open
                    self.risk_manager.on_position_closed(notional)
            else:
                all_success = False
                logger.warning(f"Failed to close position {position.ticket}: {result.comment}")

        if closed_tickets:
            self.stats['executed'] += 1
            execution = OrderExecution(
                signal=signal,
                executed=True,
                ticket=closed_tickets[0] if len(closed_tickets) == 1 else 0,
                volume=total_volume,
                error_message="" if all_success else "Some positions failed to close"
            )
            logger.info(f"Closed {len(closed_tickets)} positions, total volume: {total_volume}")
        else:
            self.stats['failed'] += 1
            execution = OrderExecution(
                signal=signal,
                executed=False,
                error_message="Failed to close any positions"
            )

        self.executions.append(execution)
        return execution

    def _validate_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Validate a trading signal.

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check confidence
        if signal.confidence < self.min_confidence:
            return False, f"Confidence {signal.confidence:.2f} below minimum {self.min_confidence}"

        # Check broker connection
        if not self.broker.is_connected:
            return False, "Broker not connected"

        # Check symbol
        symbol_info = self.broker.get_symbol_info(signal.symbol)
        if symbol_info is None:
            return False, f"Symbol {signal.symbol} not available"

        # Check spread
        tick = self.broker.get_current_tick(signal.symbol)
        if tick:
            spread_pips = tick.spread / symbol_info.point / 10  # Convert to pips
            if spread_pips > self.max_spread_pips:
                return False, f"Spread {spread_pips:.1f} pips exceeds maximum {self.max_spread_pips}"

        # Check trading allowed
        account = self.broker.get_account_info()
        if account and not account.trade_allowed:
            return False, "Trading not allowed on account"

        # Check with risk manager
        if self.risk_manager:
            if not self.risk_manager.should_take_trade():
                return False, "Risk manager rejected trade"

        return True, ""

    def _calculate_position_params(
        self,
        signal: TradingSignal,
        symbol_info: 'SymbolInfo',
        tick: 'TickInfo'
    ) -> Tuple[float, float, float]:
        """
        Calculate position size, SL, and TP.

        Returns:
            Tuple of (volume, sl_price, tp_price)
        """
        # Get risk parameters
        risk_percent = signal.risk_percent or self.default_risk_percent
        sl_pips = signal.stop_loss_pips or self.default_sl_pips
        tp_pips = signal.take_profit_pips or self.default_tp_pips

        # Get account info
        account = self.broker.get_account_info()
        if account is None:
            return 0.0, 0.0, 0.0

        # Calculate pip value
        pip_size = symbol_info.point * 10  # 1 pip = 10 points for 5-digit brokers

        # Determine entry price and SL/TP
        if signal.signal_type == SignalType.BUY:
            entry_price = tick.ask
            sl_price = entry_price - (sl_pips * pip_size)
            tp_price = entry_price + (tp_pips * pip_size)
        else:  # SELL
            entry_price = tick.bid
            sl_price = entry_price + (sl_pips * pip_size)
            tp_price = entry_price - (tp_pips * pip_size)

        # Calculate position size based on risk
        if self.risk_manager:
            volume = self.risk_manager.calculate_position_size(
                account_balance=account.balance,
                risk_percent=risk_percent,
                stop_loss_pips=sl_pips,
                pip_value=self._get_pip_value(signal.symbol, symbol_info)
            )
        else:
            # Simple position sizing: risk_amount / (sl_pips * pip_value)
            risk_amount = account.balance * risk_percent
            pip_value = self._get_pip_value(signal.symbol, symbol_info)
            if pip_value > 0 and sl_pips > 0:
                volume = risk_amount / (sl_pips * pip_value)
            else:
                volume = symbol_info.volume_min

        # Clamp volume to valid range
        volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))

        # Normalize to volume step
        volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
        volume = round(volume, 2)

        # Round SL/TP to proper digits
        sl_price = round(sl_price, symbol_info.digits)
        tp_price = round(tp_price, symbol_info.digits)

        return volume, sl_price, tp_price

    def _get_pip_value(self, symbol: str, symbol_info: 'SymbolInfo') -> float:
        """Calculate pip value for 1 lot."""
        # For forex pairs, pip value depends on quote currency
        # This is a simplified calculation
        contract_size = symbol_info.trade_contract_size
        pip_size = symbol_info.point * 10

        # Standard lot pip value (approximate)
        if symbol.endswith('USD'):
            # Direct pairs (EURUSD, GBPUSD, etc.)
            pip_value = contract_size * pip_size
        elif symbol.startswith('USD'):
            # Inverse pairs (USDJPY, USDCHF, etc.)
            tick = self.broker.get_current_tick(symbol)
            if tick and tick.bid > 0:
                pip_value = (contract_size * pip_size) / tick.bid
            else:
                pip_value = 10.0  # Default fallback
        else:
            # Cross pairs - use tick value from broker
            pip_value = symbol_info.trade_tick_value * 10  # Convert to pip

        return pip_value

    def close_all_positions(self, symbol: Optional[str] = None) -> List[OrderExecution]:
        """Close all positions."""
        signal = TradingSignal(
            signal_type=SignalType.CLOSE,
            symbol=symbol or "ALL",
            source="order_manager"
        )

        results = self.broker.close_all_positions(symbol=symbol)

        executions = []
        for result in results:
            execution = OrderExecution(
                signal=signal,
                executed=result.success,
                ticket=result.ticket,
                error_message=result.comment if not result.success else ""
            )
            executions.append(execution)
            self.executions.append(execution)

        return executions

    def get_pending_orders(self) -> List[OrderExecution]:
        """Get executions that are pending (for async systems)."""
        # In this implementation, all orders are synchronous
        return []

    def get_recent_executions(self, n: int = 10) -> List[OrderExecution]:
        """Get the n most recent executions."""
        return self.executions[-n:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            **self.stats,
            'success_rate': (
                self.stats['executed'] / max(1, self.stats['total_signals'] - self.stats['rejected'])
            ),
            'rejection_rate': (
                self.stats['rejected'] / max(1, self.stats['total_signals'])
            )
        }

    def reset_statistics(self):
        """Reset execution statistics."""
        self.stats = {
            'total_signals': 0,
            'executed': 0,
            'rejected': 0,
            'failed': 0
        }
        self.executions.clear()
