"""
Leap Trading System - MT5 Broker Gateway
Provides a clean interface to MetaTrader 5 for live trading operations.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
import time

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

logger = logging.getLogger(__name__)


class OrderType(IntEnum):
    """Order types matching MT5 constants."""
    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5


class TradeAction(IntEnum):
    """Trade action types."""
    DEAL = 1  # Market order
    PENDING = 5  # Pending order
    SLTP = 6  # Modify SL/TP
    MODIFY = 7  # Modify pending
    REMOVE = 8  # Remove pending
    CLOSE_BY = 10  # Close by opposite


@dataclass
class AccountInfo:
    """MT5 account information."""
    login: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float  # Percentage
    leverage: int
    currency: str
    server: str
    company: str
    trade_allowed: bool

    @classmethod
    def from_mt5(cls, info) -> 'AccountInfo':
        """Create from MT5 account_info result."""
        return cls(
            login=info.login,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            margin_level=info.margin_level if info.margin_level else 0.0,
            leverage=info.leverage,
            currency=info.currency,
            server=info.server,
            company=info.company,
            trade_allowed=info.trade_allowed
        )


@dataclass
class SymbolInfo:
    """MT5 symbol information."""
    name: str
    point: float  # Minimum price change
    digits: int  # Price decimal places
    spread: int  # Current spread in points
    volume_min: float  # Minimum lot size
    volume_max: float  # Maximum lot size
    volume_step: float  # Lot size step
    trade_contract_size: float  # Contract size
    trade_tick_value: float  # Tick value in account currency
    trade_tick_size: float  # Minimum tick size
    stops_level: int  # Minimum stop level in points
    freeze_level: int  # Freeze level in points
    filling_mode: int  # Filling mode flags

    @classmethod
    def from_mt5(cls, info) -> 'SymbolInfo':
        """Create from MT5 symbol_info result."""
        return cls(
            name=info.name,
            point=info.point,
            digits=info.digits,
            spread=info.spread,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            trade_contract_size=info.trade_contract_size,
            trade_tick_value=info.trade_tick_value,
            trade_tick_size=info.trade_tick_size,
            stops_level=info.trade_stops_level,
            freeze_level=info.trade_freeze_level,
            filling_mode=info.filling_mode
        )


@dataclass
class TickInfo:
    """Current market tick data."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    time: datetime

    @property
    def spread(self) -> float:
        """Calculate spread."""
        return self.ask - self.bid

    @classmethod
    def from_mt5(cls, tick, symbol: str) -> 'TickInfo':
        """Create from MT5 symbol_info_tick result."""
        return cls(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            volume=tick.volume,
            time=datetime.fromtimestamp(tick.time)
        )


@dataclass
class MT5Position:
    """Represents an open MT5 position."""
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    swap: float
    commission: float
    magic: int
    comment: str
    time: datetime

    @property
    def is_long(self) -> bool:
        return self.type == 0

    @property
    def is_short(self) -> bool:
        return self.type == 1

    @property
    def unrealized_pnl(self) -> float:
        return self.profit + self.swap + self.commission

    @classmethod
    def from_mt5(cls, pos) -> 'MT5Position':
        """Create from MT5 positions_get result."""
        return cls(
            ticket=pos.ticket,
            symbol=pos.symbol,
            type=pos.type,
            volume=pos.volume,
            price_open=pos.price_open,
            price_current=pos.price_current,
            sl=pos.sl,
            tp=pos.tp,
            profit=pos.profit,
            swap=pos.swap,
            commission=pos.commission if hasattr(pos, 'commission') else 0.0,
            magic=pos.magic,
            comment=pos.comment,
            time=datetime.fromtimestamp(pos.time)
        )


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    ticket: int = 0
    volume: float = 0.0
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    comment: str = ""
    retcode: int = 0
    request_id: int = 0

    @classmethod
    def from_mt5(cls, result) -> 'OrderResult':
        """Create from MT5 order_send result."""
        return cls(
            success=result.retcode == 10009,  # TRADE_RETCODE_DONE
            ticket=result.order,
            volume=result.volume,
            price=result.price,
            bid=result.bid,
            ask=result.ask,
            comment=result.comment,
            retcode=result.retcode,
            request_id=result.request_id
        )

    @classmethod
    def error(cls, message: str, retcode: int = -1) -> 'OrderResult':
        """Create error result."""
        return cls(success=False, comment=message, retcode=retcode)


@dataclass
class TradeHistory:
    """Historical trade record."""
    ticket: int
    symbol: str
    type: int
    volume: float
    price: float
    profit: float
    commission: float
    swap: float
    time: datetime
    comment: str

    @classmethod
    def from_mt5(cls, deal) -> 'TradeHistory':
        """Create from MT5 history_deals_get result."""
        return cls(
            ticket=deal.ticket,
            symbol=deal.symbol,
            type=deal.type,
            volume=deal.volume,
            price=deal.price,
            profit=deal.profit,
            commission=deal.commission,
            swap=deal.swap,
            time=datetime.fromtimestamp(deal.time),
            comment=deal.comment
        )


class MT5BrokerGateway:
    """
    Gateway for MetaTrader 5 broker operations.

    Provides a clean, Pythonic interface for:
    - Account information and monitoring
    - Position management
    - Order execution
    - Market data streaming
    - Trade history retrieval
    """

    # Retcode descriptions
    RETCODES = {
        10004: "Requote",
        10006: "Request rejected",
        10007: "Request canceled by trader",
        10008: "Order placed",
        10009: "Request completed",
        10010: "Only part of the request was completed",
        10011: "Request processing error",
        10012: "Request canceled by timeout",
        10013: "Invalid request",
        10014: "Invalid volume in the request",
        10015: "Invalid price in the request",
        10016: "Invalid stops in the request",
        10017: "Trade is disabled",
        10018: "Market is closed",
        10019: "There is not enough money",
        10020: "Prices changed",
        10021: "There are no quotes to process the request",
        10022: "Invalid order expiration date in the request",
        10023: "Order state changed",
        10024: "Too frequent requests",
        10025: "No changes in request",
        10026: "Autotrading disabled by server",
        10027: "Autotrading disabled by client terminal",
        10028: "Request locked for processing",
        10029: "Order or position frozen",
        10030: "Invalid order filling type",
        10031: "No connection with the trade server",
        10032: "Operation is allowed only for live accounts",
        10033: "The number of pending orders has reached the limit",
        10034: "The volume of orders and positions for the symbol has reached the limit",
        10035: "Incorrect or prohibited order type",
        10036: "Position with the specified POSITION_IDENTIFIER has already been closed",
        10038: "Close volume exceeds the current position volume",
        10039: "Close order already exists for the specified position",
        10040: "The number of open positions has reached the limit",
    }

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        timeout: int = 60000,
        magic_number: int = 234567,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize MT5 broker gateway.

        Args:
            login: MT5 account login (optional if already logged in)
            password: MT5 account password
            server: MT5 broker server
            timeout: Connection timeout in milliseconds
            magic_number: EA magic number for order identification
            max_retries: Maximum retry attempts for failed operations
            retry_delay: Delay between retries in seconds
        """
        self.login = login
        self.password = password
        self.server = server
        self.timeout = timeout
        self.magic_number = magic_number
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._is_connected = False
        self._symbol_cache: Dict[str, SymbolInfo] = {}

    @property
    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        return self._is_connected and MT5_AVAILABLE

    def connect(self) -> bool:
        """
        Connect to MetaTrader 5 terminal.

        Returns:
            True if connection successful
        """
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package not installed")
            return False

        # Initialize MT5
        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        # Login if credentials provided
        if self.login and self.password and self.server:
            if not mt5.login(
                login=self.login,
                password=self.password,
                server=self.server,
                timeout=self.timeout
            ):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        self._is_connected = True
        logger.info("Connected to MetaTrader 5")

        # Log account info
        account = self.get_account_info()
        if account:
            logger.info(f"Account: {account.login} @ {account.server}")
            logger.info(f"Balance: {account.balance} {account.currency}")
            logger.info(f"Leverage: 1:{account.leverage}")

        return True

    def disconnect(self):
        """Disconnect from MetaTrader 5."""
        if MT5_AVAILABLE and self._is_connected:
            mt5.shutdown()
            self._is_connected = False
            self._symbol_cache.clear()
            logger.info("Disconnected from MetaTrader 5")

    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get current account information.

        Returns:
            AccountInfo object or None if failed
        """
        if not self.is_connected:
            return None

        info = mt5.account_info()
        if info is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return None

        return AccountInfo.from_mt5(info)

    def get_symbol_info(self, symbol: str, use_cache: bool = True) -> Optional[SymbolInfo]:
        """
        Get symbol information.

        Args:
            symbol: Symbol name (e.g., 'EURUSD')
            use_cache: Use cached info if available

        Returns:
            SymbolInfo object or None if failed
        """
        if not self.is_connected:
            return None

        if use_cache and symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
            return None

        symbol_info = SymbolInfo.from_mt5(info)
        self._symbol_cache[symbol] = symbol_info
        return symbol_info

    def get_current_tick(self, symbol: str) -> Optional[TickInfo]:
        """
        Get current tick (bid/ask) for symbol.

        Args:
            symbol: Symbol name

        Returns:
            TickInfo object or None if failed
        """
        if not self.is_connected:
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}: {mt5.last_error()}")
            return None

        return TickInfo.from_mt5(tick, symbol)

    def get_positions(self, symbol: Optional[str] = None) -> List[MT5Position]:
        """
        Get open positions.

        Args:
            symbol: Filter by symbol (None for all)

        Returns:
            List of open positions
        """
        if not self.is_connected:
            return []

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        return [MT5Position.from_mt5(p) for p in positions]

    def get_position_by_ticket(self, ticket: int) -> Optional[MT5Position]:
        """
        Get specific position by ticket.

        Args:
            ticket: Position ticket number

        Returns:
            Position or None if not found
        """
        if not self.is_connected:
            return None

        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            return None

        return MT5Position.from_mt5(positions[0])

    def get_positions_count(self, symbol: Optional[str] = None) -> int:
        """
        Get count of open positions.

        Args:
            symbol: Filter by symbol (None for all)

        Returns:
            Number of open positions
        """
        if not self.is_connected:
            return 0

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
            return len(positions) if positions else 0

        return mt5.positions_total()

    def send_market_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        deviation: int = 20,
        comment: str = "Leap AutoTrader"
    ) -> OrderResult:
        """
        Send a market order.

        Args:
            symbol: Symbol to trade
            order_type: OrderType.BUY or OrderType.SELL
            volume: Lot size
            sl: Stop loss price (optional)
            tp: Take profit price (optional)
            deviation: Maximum price deviation in points
            comment: Order comment

        Returns:
            OrderResult with execution details
        """
        if not self.is_connected:
            return OrderResult.error("Not connected to MT5")

        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return OrderResult.error(f"Failed to get symbol info for {symbol}")

        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            return OrderResult.error(f"Failed to select symbol {symbol}")

        # Get current price
        tick = self.get_current_tick(symbol)
        if tick is None:
            return OrderResult.error(f"Failed to get tick for {symbol}")

        # Set price based on order type
        if order_type == OrderType.BUY:
            price = tick.ask
            mt5_type = mt5.ORDER_TYPE_BUY
        elif order_type == OrderType.SELL:
            price = tick.bid
            mt5_type = mt5.ORDER_TYPE_SELL
        else:
            return OrderResult.error(f"Invalid order type for market order: {order_type}")

        # Validate volume
        volume = self._normalize_volume(volume, symbol_info)
        if volume < symbol_info.volume_min:
            return OrderResult.error(
                f"Volume {volume} below minimum {symbol_info.volume_min}"
            )

        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_type,
            "price": price,
            "deviation": deviation,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol_info),
        }

        # Add SL/TP if provided
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        # Execute with retry
        return self._execute_order(request)

    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        deviation: int = 20,
        comment: str = "Leap AutoTrader Close"
    ) -> OrderResult:
        """
        Close a position.

        Args:
            ticket: Position ticket to close
            volume: Volume to close (None for full close)
            deviation: Maximum price deviation in points
            comment: Order comment

        Returns:
            OrderResult with execution details
        """
        if not self.is_connected:
            return OrderResult.error("Not connected to MT5")

        # Get position
        position = self.get_position_by_ticket(ticket)
        if position is None:
            return OrderResult.error(f"Position {ticket} not found")

        # Get symbol info
        symbol_info = self.get_symbol_info(position.symbol)
        if symbol_info is None:
            return OrderResult.error(f"Failed to get symbol info for {position.symbol}")

        # Get current price
        tick = self.get_current_tick(position.symbol)
        if tick is None:
            return OrderResult.error(f"Failed to get tick for {position.symbol}")

        # Determine close type and price
        if position.is_long:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        # Volume to close
        close_volume = volume if volume else position.volume
        close_volume = self._normalize_volume(close_volume, symbol_info)

        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": close_volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": deviation,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol_info),
        }

        return self._execute_order(request)

    def close_all_positions(
        self,
        symbol: Optional[str] = None,
        deviation: int = 20
    ) -> List[OrderResult]:
        """
        Close all positions.

        Args:
            symbol: Close only positions for this symbol (None for all)
            deviation: Maximum price deviation in points

        Returns:
            List of OrderResults
        """
        positions = self.get_positions(symbol)
        results = []

        for position in positions:
            result = self.close_position(
                ticket=position.ticket,
                deviation=deviation,
                comment="Leap AutoTrader Close All"
            )
            results.append(result)

            if not result.success:
                logger.warning(f"Failed to close position {position.ticket}: {result.comment}")

        return results

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> OrderResult:
        """
        Modify SL/TP of a position.

        Args:
            ticket: Position ticket
            sl: New stop loss (None to keep current)
            tp: New take profit (None to keep current)

        Returns:
            OrderResult with execution details
        """
        if not self.is_connected:
            return OrderResult.error("Not connected to MT5")

        # Get position
        position = self.get_position_by_ticket(ticket)
        if position is None:
            return OrderResult.error(f"Position {ticket} not found")

        # Build request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
        }

        return self._execute_order(request)

    def check_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> Tuple[bool, str, float, float]:
        """
        Check if an order would be valid before sending.

        Args:
            symbol: Symbol to trade
            order_type: Order type
            volume: Lot size
            sl: Stop loss price
            tp: Take profit price

        Returns:
            Tuple of (is_valid, message, balance, equity)
        """
        if not self.is_connected:
            return False, "Not connected to MT5", 0.0, 0.0

        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return False, f"Failed to get symbol info for {symbol}", 0.0, 0.0

        # Get current price
        tick = self.get_current_tick(symbol)
        if tick is None:
            return False, f"Failed to get tick for {symbol}", 0.0, 0.0

        # Set price based on order type
        if order_type == OrderType.BUY:
            price = tick.ask
            mt5_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            mt5_type = mt5.ORDER_TYPE_SELL

        volume = self._normalize_volume(volume, symbol_info)

        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_type,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol_info),
        }

        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        # Check order
        result = mt5.order_check(request)
        if result is None:
            return False, f"Order check failed: {mt5.last_error()}", 0.0, 0.0

        is_valid = result.retcode == 0
        message = self.RETCODES.get(result.retcode, f"Unknown error: {result.retcode}")

        if is_valid:
            message = "Order valid"

        return is_valid, message, result.balance, result.equity

    def get_trade_history(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> List[TradeHistory]:
        """
        Get trade history.

        Args:
            from_date: Start date (default: 30 days ago)
            to_date: End date (default: now)
            symbol: Filter by symbol

        Returns:
            List of historical trades
        """
        if not self.is_connected:
            return []

        if from_date is None:
            from_date = datetime.now() - timedelta(days=30)
        if to_date is None:
            to_date = datetime.now()

        from datetime import timedelta

        if symbol:
            deals = mt5.history_deals_get(from_date, to_date, symbol=symbol)
        else:
            deals = mt5.history_deals_get(from_date, to_date)

        if deals is None:
            return []

        return [TradeHistory.from_mt5(d) for d in deals]

    def calculate_profit(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        open_price: float,
        close_price: float
    ) -> float:
        """
        Calculate theoretical profit for a trade.

        Args:
            symbol: Symbol
            order_type: BUY or SELL
            volume: Lot size
            open_price: Entry price
            close_price: Exit price

        Returns:
            Profit in account currency
        """
        if not self.is_connected:
            return 0.0

        if order_type == OrderType.BUY:
            mt5_type = mt5.ORDER_TYPE_BUY
        else:
            mt5_type = mt5.ORDER_TYPE_SELL

        profit = mt5.order_calc_profit(mt5_type, symbol, volume, open_price, close_price)
        return profit if profit else 0.0

    def calculate_margin(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float
    ) -> Optional[float]:
        """
        Calculate required margin for a trade.

        Args:
            symbol: Symbol
            order_type: BUY or SELL
            volume: Lot size

        Returns:
            Required margin in account currency
        """
        if not self.is_connected:
            return None

        tick = self.get_current_tick(symbol)
        if tick is None:
            return None

        if order_type == OrderType.BUY:
            mt5_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            mt5_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        margin = mt5.order_calc_margin(mt5_type, symbol, volume, price)
        return margin

    def _normalize_volume(self, volume: float, symbol_info: SymbolInfo) -> float:
        """Normalize volume to valid lot size."""
        step = symbol_info.volume_step
        normalized = round(volume / step) * step
        normalized = max(symbol_info.volume_min, min(normalized, symbol_info.volume_max))
        return round(normalized, 2)

    def _get_filling_mode(self, symbol_info: SymbolInfo) -> int:
        """Get appropriate filling mode for symbol."""
        # Check filling modes in order of preference
        if symbol_info.filling_mode & mt5.SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        elif symbol_info.filling_mode & mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN

    def _execute_order(self, request: dict) -> OrderResult:
        """Execute order with retry logic."""
        for attempt in range(self.max_retries):
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                logger.warning(f"Order send failed (attempt {attempt + 1}): {error}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue

            order_result = OrderResult.from_mt5(result)

            if order_result.success:
                logger.info(
                    f"Order executed: {request.get('type')} {request.get('symbol')} "
                    f"{request.get('volume')} @ {order_result.price} "
                    f"(ticket: {order_result.ticket})"
                )
                return order_result

            # Check if retryable error
            if order_result.retcode in [10004, 10020, 10021]:  # Requote, prices changed, no quotes
                logger.warning(
                    f"Retryable error (attempt {attempt + 1}): "
                    f"{self.RETCODES.get(order_result.retcode, order_result.retcode)}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    # Update price for requote
                    tick = self.get_current_tick(request["symbol"])
                    if tick:
                        if request["type"] == mt5.ORDER_TYPE_BUY:
                            request["price"] = tick.ask
                        else:
                            request["price"] = tick.bid
                continue
            else:
                # Non-retryable error
                logger.error(
                    f"Order failed: {self.RETCODES.get(order_result.retcode, order_result.retcode)}"
                )
                return order_result

        return OrderResult.error(f"Max retries ({self.max_retries}) exceeded")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
