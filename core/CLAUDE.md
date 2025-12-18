# Core Module Context

The `core/` directory contains the trading system's fundamental components.

## Module Overview

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `trading_types.py` | Shared types | `Trade`, `EnvConfig`, `TradingError` hierarchy |
| `trading_env_base.py` | Abstract environment | `BaseTradingEnvironment` |
| `trading_env.py` | Backtest environment | `TradingEnvironment` |
| `live_trading_env.py` | Live environment | `LiveTradingEnvironment` |
| `risk_manager.py` | Risk management | `RiskManager`, `DynamicRiskManager` |
| `data_pipeline.py` | Data fetching | `DataPipeline`, `FeatureEngineer` |
| `auto_trader.py` | Autonomous trading | `AutoTrader` |
| `order_manager.py` | Order execution | `OrderManager` |
| `position_sync.py` | Position sync | `PositionSynchronizer` |
| `broker_interface.py` | Broker abstraction | `BrokerGateway` Protocol, `create_broker()` |
| `paper_broker.py` | Paper trading | `PaperBrokerGateway` |
| `mt5_broker.py` | MT5 integration | `MT5BrokerGateway` |

## Critical Patterns

### Broker Abstraction (broker_interface.py)
Use `BrokerGateway` Protocol for broker-agnostic code:
```python
from core.broker_interface import BrokerGateway, create_broker, PaperBrokerConfig

# Create paper broker (cross-platform)
paper_config = PaperBrokerConfig(initial_balance=10000.0, leverage=100)
broker: BrokerGateway = create_broker('paper', config=paper_config)

# Or create MT5 broker (Windows only)
from core.mt5_broker import MT5BrokerGateway
broker: BrokerGateway = MT5BrokerGateway()

# Both satisfy BrokerGateway protocol - same interface
```

All broker-dependent classes (`LiveTradingEnvironment`, `AutoTrader`, `OrderManager`, `PositionSynchronizer`) accept `BrokerGateway` type, enabling dependency injection.

### EnvConfig (trading_types.py)
Always use `EnvConfig` for environment configuration:
```python
from core.trading_types import EnvConfig
config = EnvConfig(initial_balance=50000.0, leverage=50)
# Or: EnvConfig.from_params(initial_balance=50000.0)
```

### TradingError Hierarchy (trading_types.py)
Use specific exception types:
- `TradingError` - Base for all trading errors
- `InsufficientFundsError` - Balance insufficient
- `OrderRejectedError` - Order rejected
- `PositionError` - Position-related errors
- `BrokerConnectionError` - Connection issues
- `RiskLimitExceededError` - Risk limits exceeded

### Position Sizing
1. Use `RiskManager.calculate_position_size()` if available
2. Fall back to `utils/position_sizing.py` utilities

### Environment Inheritance
```
BaseTradingEnvironment (abstract)
├── TradingEnvironment (backtest/training)
└── LiveTradingEnvironment (live trading - delegates to BrokerGateway)
```

## Common Gotchas

1. **MT5 Windows-only**: `mt5_broker.py` only works on Windows. Use `PaperBrokerGateway` for cross-platform testing.
2. **Paper mode**: Always use `--paper` flag when testing `autotrade` command
3. **Broker abstraction**: `LiveTradingEnvironment` no longer has internal paper trading logic. Paper trading is now handled by `PaperBrokerGateway`.
4. **Risk validation**: Call `RiskManager.should_take_trade()` with all 4 params:
   `entry_price`, `stop_loss_price`, `take_profit_price`, `direction`
