"""
Leap Trading System - Broker Interface Protocol

Defines the abstract interface for broker implementations.
All broker implementations (MT5, Paper, etc.) should satisfy this Protocol.

See ADR-0010 for design rationale.
"""

import logging
from typing import Protocol, Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class OrderType(IntEnum):
    """Order types for market orders."""
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


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AccountInfo:
    """Broker account information."""
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
    def create_paper(
        cls,
        balance: float = 10000.0,
        leverage: int = 100,
        currency: str = "USD"
    ) -> 'AccountInfo':
        """Create account info for paper trading."""
        return cls(
            login=0,
            balance=balance,
            equity=balance,
            margin=0.0,
            free_margin=balance,
            margin_level=0.0,
            leverage=leverage,
            currency=currency,
            server="PaperTrading",
            company="Leap Trading System",
            trade_allowed=True
        )


@dataclass
class SymbolInfo:
    """Trading symbol information."""
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
    def create_forex_default(cls, name: str) -> 'SymbolInfo':
        """Create default forex symbol info."""
        # Determine digits based on symbol name
        is_jpy_pair = 'JPY' in name.upper()
        digits = 3 if is_jpy_pair else 5
        point = 0.001 if is_jpy_pair else 0.00001

        return cls(
            name=name,
            point=point,
            digits=digits,
            spread=15,  # 1.5 pips default
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_contract_size=100000.0,  # Standard lot
            trade_tick_value=10.0 if not is_jpy_pair else 7.5,
            trade_tick_size=point,
            stops_level=0,
            freeze_level=0,
            filling_mode=1  # IOC
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

    @property
    def mid(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2


@dataclass
class BrokerPosition:
    """
    Represents an open position.

    Note: Named BrokerPosition to avoid confusion with core.trading_types.Position.
    This represents the broker's view of a position.
    """
    ticket: int
    symbol: str
    type: int  # 0=BUY/LONG, 1=SELL/SHORT
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
        """Check if this is a long position."""
        return self.type == 0

    @property
    def is_short(self) -> bool:
        """Check if this is a short position."""
        return self.type == 1

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized PnL including swap and commission."""
        return self.profit + self.swap + self.commission


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
    def error(cls, message: str, retcode: int = -1) -> 'OrderResult':
        """Create error result."""
        return cls(success=False, comment=message, retcode=retcode)

    @classmethod
    def success_result(
        cls,
        ticket: int,
        volume: float,
        price: float,
        bid: float = 0.0,
        ask: float = 0.0,
        comment: str = ""
    ) -> 'OrderResult':
        """Create success result."""
        return cls(
            success=True,
            ticket=ticket,
            volume=volume,
            price=price,
            bid=bid,
            ask=ask,
            comment=comment,
            retcode=10009  # Standard success code
        )


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


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BrokerConfig:
    """Base configuration for broker implementations."""
    magic_number: int = 234567
    max_retries: int = 3
    retry_delay: float = 1.0
    default_deviation: int = 20


@dataclass
class PaperBrokerConfig(BrokerConfig):
    """Configuration for paper trading simulation."""
    initial_balance: float = 10000.0
    leverage: int = 100
    currency: str = "USD"
    default_spread_pips: float = 1.5
    default_slippage_pips: float = 0.5
    commission_per_lot: float = 7.0  # $7 per lot round trip

    # Optional: use real market data from another broker
    use_real_prices: bool = False


# =============================================================================
# Protocol Definition
# =============================================================================

class BrokerGateway(Protocol):
    """
    Protocol defining the broker interface.

    All broker implementations must satisfy this interface.
    Uses structural subtyping - no inheritance required.

    Example:
        >>> broker: BrokerGateway = MT5BrokerGateway(login=123)
        >>> broker: BrokerGateway = PaperBrokerGateway()
    """

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Check if broker is connected and ready."""
        ...

    @property
    def magic_number(self) -> int:
        """Get the magic number for order identification."""
        ...

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Connect to the broker.

        Returns:
            True if connection successful
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from the broker."""
        ...

    # -------------------------------------------------------------------------
    # Account Information
    # -------------------------------------------------------------------------

    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get current account information.

        Returns:
            AccountInfo object or None if failed
        """
        ...

    # -------------------------------------------------------------------------
    # Symbol/Market Data
    # -------------------------------------------------------------------------

    def get_symbol_info(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[SymbolInfo]:
        """
        Get symbol information.

        Args:
            symbol: Symbol name (e.g., 'EURUSD')
            use_cache: Use cached info if available

        Returns:
            SymbolInfo object or None if failed
        """
        ...

    def get_current_tick(self, symbol: str) -> Optional[TickInfo]:
        """
        Get current tick (bid/ask) for symbol.

        Args:
            symbol: Symbol name

        Returns:
            TickInfo object or None if failed
        """
        ...

    # -------------------------------------------------------------------------
    # Position Management
    # -------------------------------------------------------------------------

    def get_positions(self, symbol: Optional[str] = None) -> List[BrokerPosition]:
        """
        Get open positions.

        Args:
            symbol: Filter by symbol (None for all)

        Returns:
            List of open positions
        """
        ...

    def get_position_by_ticket(self, ticket: int) -> Optional[BrokerPosition]:
        """
        Get specific position by ticket.

        Args:
            ticket: Position ticket number

        Returns:
            Position or None if not found
        """
        ...

    def get_positions_count(self, symbol: Optional[str] = None) -> int:
        """
        Get count of open positions.

        Args:
            symbol: Filter by symbol (None for all)

        Returns:
            Number of open positions
        """
        ...

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
        comment: str = ""
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
        ...

    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        deviation: int = 20,
        comment: str = ""
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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------

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
        ...


# =============================================================================
# Factory Function
# =============================================================================

def create_broker(
    broker_type: str = 'paper',
    config: Optional[Union[BrokerConfig, PaperBrokerConfig, Dict[str, Any]]] = None,
    **kwargs
) -> BrokerGateway:
    """
    Factory function to create broker instances.

    Args:
        broker_type: Type of broker ('mt5', 'paper')
        config: Broker configuration object or dict
        **kwargs: Additional arguments passed to broker constructor

    Returns:
        BrokerGateway implementation

    Example:
        >>> broker = create_broker('paper', initial_balance=50000.0)
        >>> broker = create_broker('mt5', login=123, password='secret', server='Demo')
    """
    if broker_type.lower() == 'mt5':
        from core.mt5_broker import MT5BrokerGateway

        if isinstance(config, dict):
            return MT5BrokerGateway(**config, **kwargs)
        elif config is not None:
            return MT5BrokerGateway(
                magic_number=config.magic_number,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay,
                **kwargs
            )
        else:
            return MT5BrokerGateway(**kwargs)

    elif broker_type.lower() == 'paper':
        from core.paper_broker import PaperBrokerGateway

        if isinstance(config, PaperBrokerConfig):
            return PaperBrokerGateway(config=config, **kwargs)
        elif isinstance(config, dict):
            paper_config = PaperBrokerConfig(**config)
            return PaperBrokerGateway(config=paper_config, **kwargs)
        else:
            return PaperBrokerGateway(**kwargs)

    else:
        raise ValueError(f"Unknown broker type: {broker_type}. Supported: 'mt5', 'paper'")


# =============================================================================
# Type Aliases for backward compatibility
# =============================================================================

# Alias for MT5Position to maintain compatibility with existing code
MT5Position = BrokerPosition


__all__ = [
    # Protocol
    'BrokerGateway',

    # Enums
    'OrderType',
    'TradeAction',

    # Data classes
    'AccountInfo',
    'SymbolInfo',
    'TickInfo',
    'BrokerPosition',
    'MT5Position',  # Alias for backward compatibility
    'OrderResult',
    'TradeHistory',

    # Configuration
    'BrokerConfig',
    'PaperBrokerConfig',

    # Factory
    'create_broker',
]
