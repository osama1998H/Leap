"""
Leap Trading System - Paper Broker Gateway

Implements the BrokerGateway protocol for simulated paper trading.
Allows cross-platform testing and development without requiring MT5.

See ADR-0010 for design rationale.
"""

import logging
import threading
import time
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from core.broker_interface import (
    BrokerGateway,
    AccountInfo,
    SymbolInfo,
    TickInfo,
    BrokerPosition,
    OrderResult,
    TradeHistory,
    OrderType,
    PaperBrokerConfig,
)
from utils.pnl_calculator import calculate_unrealized_pnl

logger = logging.getLogger(__name__)


# =============================================================================
# Price Provider Protocol
# =============================================================================

class PriceProvider:
    """
    Base class for providing price data to paper broker.

    Subclass this to implement different price sources.
    """

    def get_tick(self, symbol: str) -> Optional[TickInfo]:
        """Get current tick for symbol."""
        raise NotImplementedError

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get symbol information."""
        raise NotImplementedError


class DefaultPriceProvider(PriceProvider):
    """
    Default price provider using synthetic/static prices.

    Useful for testing and when no real market data is available.
    """

    # Default prices for common forex pairs
    DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
        'EURUSD': (1.0850, 1.0852),
        'GBPUSD': (1.2650, 1.2653),
        'USDJPY': (149.50, 149.53),
        'USDCHF': (0.8900, 0.8903),
        'AUDUSD': (0.6550, 0.6553),
        'USDCAD': (1.3600, 1.3603),
        'NZDUSD': (0.6100, 0.6103),
        'EURJPY': (162.20, 162.25),
        'GBPJPY': (189.10, 189.16),
        'EURGBP': (0.8580, 0.8583),
    }

    def __init__(self):
        self._prices: Dict[str, Tuple[float, float]] = dict(self.DEFAULT_PRICES)
        self._last_update: Dict[str, datetime] = {}

    def set_price(self, symbol: str, bid: float, ask: float) -> None:
        """Set price for a symbol (for testing/simulation)."""
        self._prices[symbol] = (bid, ask)
        self._last_update[symbol] = datetime.now()

    def get_tick(self, symbol: str) -> Optional[TickInfo]:
        """Get current tick for symbol."""
        if symbol not in self._prices:
            # Generate synthetic price based on symbol pattern
            self._generate_synthetic_price(symbol)

        if symbol not in self._prices:
            return None

        bid, ask = self._prices[symbol]
        return TickInfo(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=(bid + ask) / 2,
            volume=1000.0,
            time=datetime.now()
        )

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get symbol information."""
        return SymbolInfo.create_forex_default(symbol)

    def _generate_synthetic_price(self, symbol: str) -> None:
        """Generate a synthetic price for unknown symbols."""
        # Try to derive from related pairs
        symbol_upper = symbol.upper()

        if 'JPY' in symbol_upper:
            # JPY pairs typically trade around 100-180
            self._prices[symbol] = (130.00, 130.03)
        else:
            # Other pairs typically near 1.0
            self._prices[symbol] = (1.0000, 1.0003)

        logger.debug(f"Generated synthetic price for {symbol}")


class MT5PriceProvider(PriceProvider):
    """
    Price provider that uses real MT5 prices.

    Requires MT5BrokerGateway to be connected.
    """

    def __init__(self, mt5_broker: 'BrokerGateway'):
        self._mt5_broker = mt5_broker

    def get_tick(self, symbol: str) -> Optional[TickInfo]:
        """Get current tick from MT5."""
        return self._mt5_broker.get_current_tick(symbol)

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get symbol info from MT5."""
        return self._mt5_broker.get_symbol_info(symbol)


# =============================================================================
# Paper Broker Gateway
# =============================================================================

class PaperBrokerGateway:
    """
    Paper trading broker implementation.

    Simulates broker behavior for testing and development.
    Thread-safe for use with concurrent trading systems.

    Features:
    - Simulated positions with SL/TP monitoring
    - Configurable spread, slippage, and commissions
    - Optional real price feed from MT5
    - Complete BrokerGateway protocol implementation

    Example:
        >>> broker = PaperBrokerGateway(
        ...     config=PaperBrokerConfig(initial_balance=50000.0, leverage=100)
        ... )
        >>> broker.connect()
        >>> result = broker.send_market_order('EURUSD', OrderType.BUY, 0.1)
    """

    def __init__(
        self,
        config: Optional[PaperBrokerConfig] = None,
        price_provider: Optional[PriceProvider] = None,
        mt5_broker: Optional['BrokerGateway'] = None
    ):
        """
        Initialize paper broker gateway.

        Args:
            config: Paper broker configuration
            price_provider: Custom price provider (optional)
            mt5_broker: MT5 broker for real prices (optional, used if config.use_real_prices)
        """
        self.config = config or PaperBrokerConfig()

        # Set up price provider
        if price_provider:
            self._price_provider = price_provider
        elif mt5_broker and self.config.use_real_prices:
            self._price_provider = MT5PriceProvider(mt5_broker)
        else:
            self._price_provider = DefaultPriceProvider()

        # State
        self._is_connected = False
        self._balance = self.config.initial_balance
        self._positions: Dict[int, BrokerPosition] = {}
        self._next_ticket = 1000
        self._symbol_cache: Dict[str, SymbolInfo] = {}
        self._trade_history: List[TradeHistory] = []

        # Thread safety
        self._lock = threading.Lock()

        # Position monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._is_connected

    @property
    def magic_number(self) -> int:
        """Get magic number for order identification."""
        return self.config.magic_number

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to paper trading (always succeeds)."""
        self._is_connected = True
        logger.info("Connected to Paper Trading broker")

        # Start position monitoring thread
        self._start_position_monitoring()

        return True

    def disconnect(self) -> None:
        """Disconnect from paper trading."""
        self._stop_position_monitoring()
        self._is_connected = False
        logger.info("Disconnected from Paper Trading broker")

    # -------------------------------------------------------------------------
    # Account Information
    # -------------------------------------------------------------------------

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get current account information."""
        if not self._is_connected:
            return None

        with self._lock:
            # Calculate margin used
            margin_used = self._calculate_total_margin()

            # Calculate equity (balance + unrealized PnL)
            unrealized_pnl = self._calculate_total_unrealized_pnl()
            equity = self._balance + unrealized_pnl

            free_margin = equity - margin_used
            margin_level = (equity / margin_used * 100) if margin_used > 0 else 0.0

            return AccountInfo(
                login=0,
                balance=self._balance,
                equity=equity,
                margin=margin_used,
                free_margin=free_margin,
                margin_level=margin_level,
                leverage=self.config.leverage,
                currency=self.config.currency,
                server="PaperTrading",
                company="Leap Trading System",
                trade_allowed=True
            )

    # -------------------------------------------------------------------------
    # Symbol/Market Data
    # -------------------------------------------------------------------------

    def get_symbol_info(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[SymbolInfo]:
        """Get symbol information."""
        if not self._is_connected:
            return None

        if use_cache and symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        symbol_info = self._price_provider.get_symbol_info(symbol)
        if symbol_info:
            self._symbol_cache[symbol] = symbol_info

        return symbol_info

    def get_current_tick(self, symbol: str) -> Optional[TickInfo]:
        """Get current tick for symbol."""
        if not self._is_connected:
            return None

        tick = self._price_provider.get_tick(symbol)
        if tick is None:
            return None

        # Add slippage simulation (small random variation)
        # In production, this could be more sophisticated
        return tick

    # -------------------------------------------------------------------------
    # Position Management
    # -------------------------------------------------------------------------

    def get_positions(self, symbol: Optional[str] = None) -> List[BrokerPosition]:
        """Get open positions."""
        if not self._is_connected:
            return []

        with self._lock:
            if symbol:
                return [p for p in self._positions.values() if p.symbol == symbol]
            return list(self._positions.values())

    def get_position_by_ticket(self, ticket: int) -> Optional[BrokerPosition]:
        """Get specific position by ticket."""
        if not self._is_connected:
            return None

        with self._lock:
            return self._positions.get(ticket)

    def get_positions_count(self, symbol: Optional[str] = None) -> int:
        """Get count of open positions."""
        return len(self.get_positions(symbol))

    # -------------------------------------------------------------------------
    # Order Execution
    # -------------------------------------------------------------------------

    def send_market_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        deviation: int = 20,
        comment: str = "Leap PaperTrading"
    ) -> OrderResult:
        """Send a market order."""
        if not self._is_connected:
            return OrderResult.error("Not connected to broker")

        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return OrderResult.error(f"Failed to get symbol info for {symbol}")

        # Get current tick
        tick = self.get_current_tick(symbol)
        if tick is None:
            return OrderResult.error(f"Failed to get tick for {symbol}")

        # Validate volume
        volume = self._normalize_volume(volume, symbol_info)
        if volume < symbol_info.volume_min:
            return OrderResult.error(
                f"Volume {volume} below minimum {symbol_info.volume_min}"
            )

        # Determine price based on order type
        if order_type == OrderType.BUY:
            price = tick.ask
            position_type = 0  # LONG
        elif order_type == OrderType.SELL:
            price = tick.bid
            position_type = 1  # SHORT
        else:
            return OrderResult.error(f"Invalid order type for market order: {order_type}")

        # Apply slippage
        slippage = symbol_info.point * self.config.default_slippage_pips * 10
        if order_type == OrderType.BUY:
            price += slippage
        else:
            price -= slippage

        # Check margin
        required_margin = self._calculate_margin_for_order(symbol, volume)
        if required_margin is None:
            return OrderResult.error("Failed to calculate margin")

        account = self.get_account_info()
        if account and account.free_margin < required_margin:
            return OrderResult.error(
                f"Insufficient margin. Required: {required_margin:.2f}, "
                f"Available: {account.free_margin:.2f}"
            )

        # Create position
        with self._lock:
            ticket = self._next_ticket
            self._next_ticket += 1

            # Calculate commission
            commission = self.config.commission_per_lot * volume / 2  # Half on open

            position = BrokerPosition(
                ticket=ticket,
                symbol=symbol,
                type=position_type,
                volume=volume,
                price_open=price,
                price_current=price,
                sl=sl or 0.0,
                tp=tp or 0.0,
                profit=0.0,
                swap=0.0,
                commission=commission,
                magic=self.config.magic_number,
                comment=comment,
                time=datetime.now()
            )

            self._positions[ticket] = position

        logger.info(
            f"Paper order executed: {'BUY' if order_type == OrderType.BUY else 'SELL'} "
            f"{symbol} {volume} @ {price:.5f} (ticket: {ticket})"
        )

        return OrderResult.success_result(
            ticket=ticket,
            volume=volume,
            price=price,
            bid=tick.bid,
            ask=tick.ask,
            comment="Paper trade executed"
        )

    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        deviation: int = 20,
        comment: str = "Leap PaperTrading Close"
    ) -> OrderResult:
        """Close a position."""
        if not self._is_connected:
            return OrderResult.error("Not connected to broker")

        with self._lock:
            position = self._positions.get(ticket)
            if position is None:
                return OrderResult.error(f"Position {ticket} not found")

            # Get current tick
            tick = self.get_current_tick(position.symbol)
            if tick is None:
                return OrderResult.error(f"Failed to get tick for {position.symbol}")

            # Determine close price
            if position.is_long:
                close_price = tick.bid
            else:
                close_price = tick.ask

            # Calculate PnL
            symbol_info = self.get_symbol_info(position.symbol)
            contract_size = symbol_info.trade_contract_size if symbol_info else 100000
            pnl = self._calculate_position_pnl(position, close_price, contract_size)

            # Add commission on close
            commission = self.config.commission_per_lot * position.volume / 2
            pnl -= commission

            # Update balance
            self._balance += pnl

            # Record in history
            self._trade_history.append(TradeHistory(
                ticket=ticket,
                symbol=position.symbol,
                type=position.type,
                volume=position.volume,
                price=close_price,
                profit=pnl,
                commission=position.commission + commission,
                swap=position.swap,
                time=datetime.now(),
                comment=comment
            ))

            # Remove position
            del self._positions[ticket]

        logger.info(
            f"Paper position closed: {ticket} @ {close_price:.5f} "
            f"(PnL: {pnl:.2f})"
        )

        return OrderResult.success_result(
            ticket=ticket,
            volume=position.volume,
            price=close_price,
            bid=tick.bid,
            ask=tick.ask,
            comment=f"Position closed. PnL: {pnl:.2f}"
        )

    def close_all_positions(
        self,
        symbol: Optional[str] = None,
        deviation: int = 20
    ) -> List[OrderResult]:
        """Close all positions."""
        positions = self.get_positions(symbol)
        results = []

        for position in positions:
            result = self.close_position(
                ticket=position.ticket,
                deviation=deviation,
                comment="Leap PaperTrading Close All"
            )
            results.append(result)

        return results

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> OrderResult:
        """Modify SL/TP of a position."""
        if not self._is_connected:
            return OrderResult.error("Not connected to broker")

        with self._lock:
            position = self._positions.get(ticket)
            if position is None:
                return OrderResult.error(f"Position {ticket} not found")

            # Update position (create new dataclass instance)
            updated = BrokerPosition(
                ticket=position.ticket,
                symbol=position.symbol,
                type=position.type,
                volume=position.volume,
                price_open=position.price_open,
                price_current=position.price_current,
                sl=sl if sl is not None else position.sl,
                tp=tp if tp is not None else position.tp,
                profit=position.profit,
                swap=position.swap,
                commission=position.commission,
                magic=position.magic,
                comment=position.comment,
                time=position.time
            )

            self._positions[ticket] = updated

        logger.info(f"Paper position {ticket} modified: SL={sl}, TP={tp}")

        return OrderResult.success_result(
            ticket=ticket,
            volume=position.volume,
            price=position.price_open,
            comment="Position modified"
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def check_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> Tuple[bool, str, float, float]:
        """Check if an order would be valid before sending."""
        if not self._is_connected:
            return False, "Not connected to broker", 0.0, 0.0

        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return False, f"Failed to get symbol info for {symbol}", 0.0, 0.0

        # Validate volume
        volume = self._normalize_volume(volume, symbol_info)
        if volume < symbol_info.volume_min:
            return False, f"Volume {volume} below minimum {symbol_info.volume_min}", 0.0, 0.0
        if volume > symbol_info.volume_max:
            return False, f"Volume {volume} above maximum {symbol_info.volume_max}", 0.0, 0.0

        # Check margin
        required_margin = self._calculate_margin_for_order(symbol, volume)
        if required_margin is None:
            return False, "Failed to calculate margin", 0.0, 0.0

        account = self.get_account_info()
        if account is None:
            return False, "Failed to get account info", 0.0, 0.0

        if account.free_margin < required_margin:
            return False, "Insufficient margin", account.balance, account.equity

        return True, "Order valid", account.balance, account.equity

    # -------------------------------------------------------------------------
    # Calculations
    # -------------------------------------------------------------------------

    def calculate_profit(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        open_price: float,
        close_price: float
    ) -> float:
        """Calculate theoretical profit for a trade."""
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return 0.0

        contract_size = symbol_info.trade_contract_size

        if order_type == OrderType.BUY:
            price_diff = close_price - open_price
        else:
            price_diff = open_price - close_price

        # PnL = price_diff * volume * contract_size
        profit = price_diff * volume * contract_size

        # For JPY pairs, adjust for point value
        if 'JPY' in symbol.upper():
            profit *= 0.01  # Adjust for 3-digit pricing

        return profit

    def calculate_margin(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float
    ) -> Optional[float]:
        """Calculate required margin for a trade."""
        return self._calculate_margin_for_order(symbol, volume)

    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------

    def get_trade_history(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> List[TradeHistory]:
        """Get trade history."""
        with self._lock:
            history = list(self._trade_history)

        # Filter by date
        if from_date:
            history = [h for h in history if h.time >= from_date]
        if to_date:
            history = [h for h in history if h.time <= to_date]

        # Filter by symbol
        if symbol:
            history = [h for h in history if h.symbol == symbol]

        return history

    # -------------------------------------------------------------------------
    # Price Management (for testing)
    # -------------------------------------------------------------------------

    def set_price(self, symbol: str, bid: float, ask: float) -> None:
        """
        Set price for a symbol (for testing).

        Only works with DefaultPriceProvider.
        """
        if isinstance(self._price_provider, DefaultPriceProvider):
            self._price_provider.set_price(symbol, bid, ask)
            # Update position current prices
            self._update_position_prices()
        else:
            logger.warning("set_price only works with DefaultPriceProvider")

    def set_balance(self, balance: float) -> None:
        """Set balance (for testing)."""
        with self._lock:
            self._balance = balance

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _normalize_volume(self, volume: float, symbol_info: SymbolInfo) -> float:
        """Normalize volume to valid lot size."""
        step = symbol_info.volume_step
        normalized = round(volume / step) * step
        normalized = max(symbol_info.volume_min, min(normalized, symbol_info.volume_max))
        return round(normalized, 2)

    def _calculate_margin_for_order(
        self,
        symbol: str,
        volume: float
    ) -> Optional[float]:
        """Calculate margin required for an order."""
        tick = self.get_current_tick(symbol)
        if tick is None:
            return None

        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return None

        contract_size = symbol_info.trade_contract_size
        price = tick.mid

        # Margin = (volume * contract_size * price) / leverage
        margin = (volume * contract_size * price) / self.config.leverage

        return margin

    def _calculate_total_margin(self) -> float:
        """Calculate total margin used by all positions."""
        total_margin = 0.0

        for position in self._positions.values():
            symbol_info = self.get_symbol_info(position.symbol)
            if symbol_info:
                contract_size = symbol_info.trade_contract_size
                margin = (position.volume * contract_size * position.price_open) / self.config.leverage
                total_margin += margin

        return total_margin

    def _calculate_position_pnl(
        self,
        position: BrokerPosition,
        current_price: float,
        contract_size: float
    ) -> float:
        """Calculate PnL for a position using centralized utility."""
        direction = 'long' if position.is_long else 'short'
        pnl = calculate_unrealized_pnl(
            entry_price=position.price_open,
            current_price=current_price,
            size=position.volume,
            direction=direction,
            contract_size=contract_size
        )

        # Adjust for JPY pairs (different pip value)
        if 'JPY' in position.symbol.upper():
            pnl *= 0.01

        return pnl

    def _calculate_total_unrealized_pnl(self) -> float:
        """Calculate total unrealized PnL for all positions."""
        total_pnl = 0.0

        for position in self._positions.values():
            tick = self._price_provider.get_tick(position.symbol)
            if tick:
                symbol_info = self.get_symbol_info(position.symbol)
                contract_size = symbol_info.trade_contract_size if symbol_info else 100000

                current_price = tick.bid if position.is_long else tick.ask
                total_pnl += self._calculate_position_pnl(position, current_price, contract_size)

        return total_pnl

    def _update_position_prices(self) -> None:
        """Update current prices for all positions."""
        with self._lock:
            for ticket, position in self._positions.items():
                tick = self._price_provider.get_tick(position.symbol)
                if tick:
                    current_price = tick.bid if position.is_long else tick.ask
                    symbol_info = self.get_symbol_info(position.symbol)
                    contract_size = symbol_info.trade_contract_size if symbol_info else 100000

                    profit = self._calculate_position_pnl(position, current_price, contract_size)

                    # Update position
                    self._positions[ticket] = BrokerPosition(
                        ticket=position.ticket,
                        symbol=position.symbol,
                        type=position.type,
                        volume=position.volume,
                        price_open=position.price_open,
                        price_current=current_price,
                        sl=position.sl,
                        tp=position.tp,
                        profit=profit,
                        swap=position.swap,
                        commission=position.commission,
                        magic=position.magic,
                        comment=position.comment,
                        time=position.time
                    )

    # -------------------------------------------------------------------------
    # Position Monitoring (SL/TP)
    # -------------------------------------------------------------------------

    def _start_position_monitoring(self) -> None:
        """Start background thread for SL/TP monitoring."""
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._position_monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.debug("Position monitoring started")

    def _stop_position_monitoring(self) -> None:
        """Stop position monitoring thread."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        logger.debug("Position monitoring stopped")

    def _position_monitor_loop(self) -> None:
        """Background loop to monitor positions for SL/TP hits."""
        while not self._stop_monitoring.is_set():
            try:
                self._check_sl_tp()
                self._update_position_prices()
            except Exception as e:
                logger.warning(f"Error in position monitor: {e}")

            # Sleep for a short interval
            self._stop_monitoring.wait(0.5)

    def _check_sl_tp(self) -> None:
        """Check all positions for SL/TP hits."""
        positions_to_close: List[Tuple[int, float, str]] = []

        with self._lock:
            for ticket, position in self._positions.items():
                tick = self._price_provider.get_tick(position.symbol)
                if tick is None:
                    continue

                current_price = tick.bid if position.is_long else tick.ask

                # Check stop loss
                if position.sl > 0:
                    if position.is_long and current_price <= position.sl:
                        positions_to_close.append((ticket, position.sl, "Stop Loss hit"))
                    elif position.is_short and current_price >= position.sl:
                        positions_to_close.append((ticket, position.sl, "Stop Loss hit"))

                # Check take profit
                if position.tp > 0:
                    if position.is_long and current_price >= position.tp:
                        positions_to_close.append((ticket, position.tp, "Take Profit hit"))
                    elif position.is_short and current_price <= position.tp:
                        positions_to_close.append((ticket, position.tp, "Take Profit hit"))

        # Close positions outside lock to avoid deadlock
        for ticket, price, reason in positions_to_close:
            logger.info(f"Paper position {ticket} closing: {reason} @ {price:.5f}")
            self.close_position(ticket, comment=reason)

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


__all__ = [
    'PaperBrokerGateway',
    'PaperBrokerConfig',
    'PriceProvider',
    'DefaultPriceProvider',
    'MT5PriceProvider',
]
