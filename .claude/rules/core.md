---
paths: core/**/*.py
---
# Core Module Development Rules

These rules apply when working on files in the `core/` directory.

---

## Module Overview

The `core/` module contains the trading system fundamentals:

| File | Responsibility |
|------|----------------|
| `trading_types.py` | Shared types: `Trade`, `EnvConfig`, `TradingError` hierarchy |
| `trading_env_base.py` | Abstract `BaseTradingEnvironment` |
| `trading_env.py` | `TradingEnvironment` for backtesting |
| `live_trading_env.py` | `LiveTradingEnvironment` for live/paper trading |
| `broker_interface.py` | `BrokerGateway` Protocol and `create_broker()` factory |
| `mt5_broker.py` | `MT5BrokerGateway` implementation |
| `paper_broker.py` | `PaperBrokerGateway` implementation |
| `strategy.py` | `TradingStrategy` ABC and implementations |
| `risk_manager.py` | `RiskManager` for position sizing and limits |
| `order_manager.py` | `OrderManager` for order execution |
| `position_sync.py` | `PositionSynchronizer` for broker sync |
| `data_pipeline.py` | Data fetching and preparation |
| `feature_engineer.py` | Technical indicator computation |
| `feature_registry.py` | Feature computation registry |
| `auto_trader.py` | `AutoTrader` orchestration |

---

## Required Patterns

### TYPE_CHECKING for Optional Runtime Dependencies

When a type hint creates an import cycle or optional dependency:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.risk_manager import RiskManager
    from models.transformer import TransformerPredictor

class OrderManager:
    def __init__(self, risk_manager: "RiskManager | None" = None):
        ...
```

### Broker Abstraction

ALL broker access MUST go through `BrokerGateway` Protocol:

```python
from core.broker_interface import BrokerGateway, create_broker

# Create broker (paper or MT5)
broker: BrokerGateway = create_broker('paper', initial_balance=10000.0)

# Use broker methods
positions = broker.get_positions()
order_result = broker.place_order(order)
```

### EnvConfig for Environment Configuration

Always use `EnvConfig` for trading environment setup:

```python
from core.trading_types import EnvConfig

# Direct instantiation
config = EnvConfig(initial_balance=50000.0, leverage=50)

# Factory method (cleaner)
config = EnvConfig.from_params(initial_balance=50000.0)

# Use in environment
env = TradingEnvironment(data=data, config=config)
```

### TradingError Hierarchy

Use specific error types for trading exceptions:

```python
from core.trading_types import (
    TradingError,           # Base class
    InsufficientFundsError, # Not enough margin/balance
    OrderRejectedError,     # Order rejected by broker
    PositionError,          # Position-related errors
    BrokerConnectionError,  # Broker connectivity issues
    DataPipelineError,      # Data fetching/processing errors
    RiskLimitExceededError, # Risk limits exceeded
)

# Raise specific errors
if balance < required:
    raise InsufficientFundsError(f"Need {required}, have {balance}")
```

### Strategy Pattern

All trading strategies MUST extend `TradingStrategy`:

```python
from core.strategy import TradingStrategy, StrategySignal

class MyStrategy(TradingStrategy):
    def generate_signal(
        self,
        market_data: MarketData,
        positions: list[Position],
        **kwargs
    ) -> StrategySignal:
        # Implement signal generation
        return StrategySignal(action=action, confidence=0.8)

    def on_trade_opened(self, trade: Trade) -> None:
        # Optional: React to trade opening
        pass

    def on_trade_closed(self, trade: Trade) -> None:
        # Optional: React to trade closing
        pass
```

---

## MT5 Windows-Conditional Code

MetaTrader 5 only works on Windows. Conditional handling:

```python
import platform

MT5_AVAILABLE = False
if platform.system() == 'Windows':
    try:
        import MetaTrader5 as mt5
        MT5_AVAILABLE = True
    except ImportError:
        pass

def require_mt5():
    if not MT5_AVAILABLE:
        raise RuntimeError("MT5 only available on Windows")
```

---

## Position Sizing Delegation

When calculating position sizes:

```python
# 1. Use RiskManager if available (preferred)
if self.risk_manager:
    size = self.risk_manager.calculate_position_size(
        entry_price=price,
        stop_loss_price=sl,
        risk_per_trade=0.02
    )
else:
    # 2. Fall back to utility functions
    from utils.position_sizing import calculate_risk_based_size
    size = calculate_risk_based_size(
        balance=balance,
        entry_price=price,
        stop_loss_price=sl,
        risk_percent=0.02
    )
```

---

## Environment Inheritance

```
BaseTradingEnvironment (ABC)
    ├── TradingEnvironment       # For backtesting
    └── LiveTradingEnvironment   # For live/paper trading
```

When modifying environments:
- Changes to shared behavior → `BaseTradingEnvironment`
- Backtest-specific → `TradingEnvironment`
- Live-specific → `LiveTradingEnvironment`

---

## Testing Core Components

```bash
# Run core tests
python -m pytest tests/test_trading_env.py -v
python -m pytest tests/test_broker_interface.py -v
python -m pytest tests/test_risk_manager.py -v
python -m pytest tests/test_strategy.py -v
```

---

## DO NOT

- Import from `cli/` (creates upward dependency)
- Create new error types outside `TradingError` hierarchy
- Bypass `BrokerGateway` for broker operations
- Use MT5 directly without platform check
- Skip `EnvConfig` for environment setup
