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
| `mt5_broker.py` | MT5 integration | `MT5BrokerGateway` |

## Critical Patterns

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
└── LiveTradingEnvironment (live trading)
```

## Common Gotchas

1. **MT5 Windows-only**: `mt5_broker.py` only works on Windows
2. **Paper mode**: Always use `--paper` flag when testing `autotrade`
3. **Risk validation**: Call `RiskManager.should_take_trade()` with all 4 params:
   `entry_price`, `stop_loss_price`, `take_profit_price`, `direction`
