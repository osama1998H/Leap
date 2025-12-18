# Leap Auto-Trader Documentation

## Overview

The Leap Auto-Trader is an autonomous trading system that integrates the Transformer price prediction model with the PPO reinforcement learning agent to execute trades automatically on MetaTrader 5 (MT5).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AUTO-TRADER SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ MT5 Broker   │    │ Order        │    │ Position     │       │
│  │ Gateway      │◄──►│ Manager      │◄──►│ Synchronizer │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                    │               │
│         └───────────────────┼────────────────────┘               │
│                             ▼                                    │
│                    ┌──────────────────┐                          │
│                    │   Auto-Trader    │                          │
│                    │   Orchestrator   │                          │
│                    └──────────────────┘                          │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Transformer  │  │  PPO Agent   │  │   Online     │          │
│  │  Predictor   │  │  (Actions)   │  │  Learning    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. MT5 Broker Gateway (`core/mt5_broker.py`)

Provides a clean Python interface to MetaTrader 5:

- **Connection Management**: `connect()`, `disconnect()`
- **Account Info**: `get_account_info()` - balance, equity, margin
- **Market Data**: `get_current_tick()`, `get_symbol_info()`
- **Position Management**: `get_positions()`, `get_position_by_ticket()`
- **Order Execution**: `send_market_order()`, `close_position()`, `modify_position()`

### 2. Order Manager (`core/order_manager.py`)

Handles trade signal execution:

- Signal validation (confidence, spread, risk limits)
- Position sizing based on risk percentage
- Stop loss / Take profit calculation
- Order execution with retry logic
- Execution logging and statistics

### 3. Position Synchronizer (`core/position_sync.py`)

Maintains state consistency with MT5:

- Real-time position tracking
- Detection of external trades (manual or other EAs)
- SL/TP hit detection
- Event callbacks for position changes

### 4. Live Trading Environment (`core/live_trading_env.py`)

Gymnasium-compatible environment for live trading:

- Real-time observation from MT5
- `open_status` flag to control position entry
- Paper trading mode for testing
- Compatible with trained PPO agent

### 5. Auto-Trader (`core/auto_trader.py`)

Main orchestrator that coordinates all components:

- Trading loop with configurable interval
- Signal generation from models
- Risk validation before execution
- Online learning adaptation triggers
- Session statistics tracking

## Configuration

### AutoTraderConfig

```python
from config.settings import AutoTraderConfig

config = AutoTraderConfig(
    # Trading settings
    symbols=['EURUSD'],
    timeframe='1h',

    # Risk settings
    risk_per_trade=0.01,      # 1% risk per trade
    max_positions=3,
    max_daily_loss=0.05,      # 5% daily loss limit
    max_drawdown=0.10,        # 10% max drawdown

    # Stop loss / Take profit
    default_sl_pips=50.0,
    default_tp_pips=100.0,

    # Trading hours (UTC)
    trading_start_hour=8,
    trading_end_hour=20,
    trading_days=[0, 1, 2, 3, 4],  # Mon-Fri

    # Online learning
    enable_online_learning=True,
    adaptation_frequency=100,  # Adapt every 100 trades
)
```

## Usage

### Command Line

```bash
# Paper trading mode (recommended for testing)
python main.py autotrade --paper --symbol EURUSD

# Live trading (requires confirmation)
python main.py autotrade --symbol EURUSD

# With specific model directory
python main.py autotrade --paper --model-dir ./saved_models --symbol GBPUSD
```

### Programmatic Usage

```python
from core.broker_interface import create_broker, PaperBrokerConfig
from core.mt5_broker import MT5BrokerGateway
from core.auto_trader import AutoTrader, AutoTraderConfig
from models.transformer import TransformerPredictor
from models.ppo_agent import PPOAgent

# Option 1: Paper trading (cross-platform)
paper_config = PaperBrokerConfig(initial_balance=10000.0, leverage=100)
broker = create_broker('paper', config=paper_config)

# Option 2: Live trading with MT5 (Windows only)
# broker = MT5BrokerGateway()

predictor = TransformerPredictor.load('./models/predictor.pt')
agent = PPOAgent.load('./models/agent.pt')

# Configure
config = AutoTraderConfig(
    symbols=['EURUSD'],
    risk_per_trade=0.01
)

# Create auto-trader (accepts any BrokerGateway implementation)
trader = AutoTrader(
    broker=broker,
    predictor=predictor,
    agent=agent,
    config=config
)

# Start trading
trader.start()

# ... trading runs in background ...

# Stop trading
trader.stop()

# Get statistics
stats = trader.get_statistics()
print(f"Total trades: {stats['session']['total_trades']}")
print(f"Win rate: {stats['session']['win_rate']*100:.1f}%")
```

## Open Status Control

The `open_status` feature allows external control over position entry:

```python
# Disable opening new positions
trader.set_open_status('EURUSD', can_open=False)

# Enable close-only mode (can only close, no new entries)
trader.set_close_only('EURUSD', close_only=True)

# Re-enable trading
trader.set_open_status('EURUSD', can_open=True)
```

This is useful for:
- Pausing during news events
- Gradual wind-down of positions
- Integration with external risk systems

## Online Learning Integration

The auto-trader supports continuous model adaptation:

1. **Trade Completion**: Each completed trade is logged
2. **Buffer Accumulation**: Trades accumulate in experience buffer
3. **Adaptation Trigger**: After N trades, adaptation is triggered
4. **Model Update**: Predictor and agent are updated with recent data

```python
config = AutoTraderConfig(
    enable_online_learning=True,
    adaptation_frequency=100,      # Trades between adaptations
    max_adaptations_per_day=10    # Limit daily adaptations
)
```

## Safety Features

### Risk Management

- **Position Limits**: Maximum concurrent positions
- **Daily Loss Limit**: Stop trading after 5% daily loss
- **Max Drawdown**: Circuit breaker at 10% drawdown
- **Risk per Trade**: Fixed percentage risk sizing

### Trading Controls

- **Paper Mode**: Default for safe testing
- **Confirmation Required**: Live mode requires explicit confirmation
- **Trading Hours**: Configurable trading windows
- **Trading Days**: Exclude weekends by default

### Error Handling

- **Connection Retry**: Automatic reconnection on disconnection
- **Order Retry**: Retry failed orders with updated prices
- **Graceful Shutdown**: Clean position sync on stop

## Platform Requirements

- **Operating System**: Windows (MT5 requirement)
- **MetaTrader 5**: Must be installed and running
- **Python Package**: `MetaTrader5` (`pip install MetaTrader5`)
- **Account**: Demo or live MT5 account

For Linux/macOS development, use paper mode with synthetic data.

## Monitoring

### Status

```python
status = trader.get_status()
# Returns: state, balance, equity, positions, session info
```

### Statistics

```python
stats = trader.get_statistics()
# Returns: detailed trading statistics
```

### Callbacks

```python
def on_trade(data):
    print(f"Trade executed: {data['execution'].ticket}")

trader.register_callback('on_trade', on_trade)
trader.register_callback('on_signal', lambda d: print(f"Signal: {d['signal']}"))
trader.register_callback('on_error', lambda d: print(f"Error: {d['error']}"))
```

## Troubleshooting

### Common Issues

1. **MT5 Not Available**
   - Ensure MetaTrader 5 is installed
   - Run on Windows
   - Install: `pip install MetaTrader5`

2. **Connection Failed**
   - Check MT5 terminal is running
   - Verify account credentials
   - Enable "Allow automated trading" in MT5

3. **Orders Rejected**
   - Check spread conditions
   - Verify sufficient margin
   - Check trading hours

4. **No Trades Executed**
   - Check model predictions (confidence threshold)
   - Verify `open_status` is True
   - Check risk manager constraints

## Best Practices

1. **Always start with paper mode** to verify behavior
2. **Use conservative risk settings** (1% or less per trade)
3. **Monitor regularly** especially during initial deployment
4. **Set appropriate trading hours** to avoid low-liquidity periods
5. **Enable online learning** for market adaptation
6. **Keep logs** for post-trading analysis
