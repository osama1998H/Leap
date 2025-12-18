# ADR 0010: Broker Interface Abstraction

## Status
Accepted

## Context

The trading system has tight coupling between components and specific broker implementations:

1. **`AutoTrader`** directly imports and uses `MT5BrokerGateway`
2. **`LiveTradingEnvironment`** contains mixed paper trading and real trading logic
3. **`OrderManager`** and **`PositionSynchronizer`** hardcode broker types
4. Testing requires Windows with MetaTrader 5 installed

This coupling:
- Makes unit testing difficult
- Prevents broker-agnostic components
- Blocks paper trading on non-Windows platforms
- Requires changing multiple files to add a new broker

## Decision

Introduce a broker abstraction layer using Python's structural subtyping (Protocol):

### 1. BrokerGateway Protocol (`core/broker_interface.py`)

```python
class BrokerGateway(Protocol):
    @property
    def is_connected(self) -> bool: ...
    @property
    def magic_number(self) -> int: ...

    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def get_account_info(self) -> Optional[AccountInfo]: ...
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]: ...
    def get_current_tick(self, symbol: str) -> Optional[TickInfo]: ...
    def get_positions(self, symbol: Optional[str] = None) -> List[BrokerPosition]: ...
    def send_market_order(...) -> OrderResult: ...
    def close_position(ticket: int) -> OrderResult: ...
    def close_all_positions(...) -> List[OrderResult]: ...
    def modify_position(ticket: int, sl: float, tp: float) -> OrderResult: ...
    def check_order(...) -> Tuple[bool, str, float, float]: ...
    def calculate_profit(...) -> float: ...
    def calculate_margin(...) -> Optional[float]: ...
    def get_trade_history(...) -> List[TradeHistory]: ...
```

### 2. Shared Data Classes

Move to `core/broker_interface.py`:
- `AccountInfo`
- `SymbolInfo`
- `TickInfo`
- `BrokerPosition`
- `OrderResult`
- `TradeHistory`
- `OrderType` enum
- `TradeAction` enum

### 3. PaperBrokerGateway (`core/paper_broker.py`)

Full implementation for paper trading:
- Internal position tracking with `Dict[int, BrokerPosition]`
- Thread-safe via `threading.Lock`
- Configurable spread, slippage, commissions
- Background SL/TP monitoring thread
- Optional real-price integration via `MT5PriceProvider`

### 4. Factory Function

```python
def create_broker(
    broker_type: str = 'paper',
    config: Optional[Union[BrokerConfig, Dict]] = None
) -> BrokerGateway:
    """
    Create broker instance.

    Args:
        broker_type: 'mt5' or 'paper'
        config: Broker configuration
    """
```

### Why Protocol over ABC?

Using `typing.Protocol` instead of `abc.ABC`:

1. **Structural subtyping**: `MT5BrokerGateway` satisfies the protocol without modification
2. **Backward compatible**: No changes to existing broker implementations
3. **Better for type checking**: mypy understands protocol compliance
4. **Duck typing**: Any class with matching methods/properties works

## Consequences

### Positive
- **Testability**: Mock brokers for unit tests
- **Platform independence**: Paper trading works on Linux/macOS
- **Separation of concerns**: Broker logic isolated from trading logic
- **Extensibility**: Add new brokers (Alpaca, Interactive Brokers) easily

### Negative
- **Additional abstraction layer**: More files to maintain
- **Data class duplication**: Some overlap with existing MT5 types (mitigated by imports)
- **Runtime duck typing**: Protocol compliance checked at runtime, not compile time

## Implementation Files

| File | Purpose |
|------|---------|
| `core/broker_interface.py` | Protocol + data classes + factory |
| `core/paper_broker.py` | PaperBrokerGateway implementation |
| `tests/test_broker_interface.py` | Comprehensive tests |

## Migration Path

1. Import data classes from `broker_interface.py` in `mt5_broker.py`
2. Type hint broker consumers with `BrokerGateway`
3. Use `create_broker()` factory in CLI commands
4. Future: Extract paper trading from `LiveTradingEnvironment`

## Alternatives Considered

### 1. Abstract Base Class
- Required modifying `MT5BrokerGateway` to inherit from ABC
- Would break backward compatibility
- Less Pythonic than Protocol

### 2. Keep inline paper trading
- Already messy in `LiveTradingEnvironment`
- Makes testing harder
- Violates single responsibility principle

### 3. External broker library
- No suitable library for MT5 + paper trading combination
- Would add heavy dependencies
- Less control over implementation
