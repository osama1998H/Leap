# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Leap is an AI-powered forex trading system combining Transformer-based price prediction with PPO reinforcement learning. Features online learning for continuous market adaptation, walk-forward optimization, and MetaTrader 5 integration for live trading.

**Key Capabilities:**
- Temporal Fusion Transformer for price prediction with uncertainty quantiles
- PPO actor-critic agent for trading decisions (HOLD, BUY, SELL, CLOSE)
- Online learning system with market regime detection
- Walk-forward optimization and Monte Carlo simulation
- Live trading via MetaTrader 5 (Windows only) with paper trading mode
- MLflow experiment tracking and model versioning

## Common Commands

```bash
# Training
python main.py train --symbol EURUSD --timeframe 1h --epochs 100 --timesteps 100000
python main.py train --config config/custom_config.json

# Backtesting
python main.py backtest --symbol EURUSD --bars 50000
python main.py backtest --symbol EURUSD --realistic --monte-carlo
python main.py walkforward --symbol EURUSD

# Live Trading
python main.py live --paper                              # Paper trading
python main.py autotrade --paper --symbol EURUSD         # Auto-trader with MT5 (Windows only)

# Evaluation
python main.py evaluate --model-dir ./saved_models

# Testing
python -m pytest tests/ -v                               # All tests
python -m pytest tests/test_integration.py -v            # Integration tests
python -m pytest tests/test_auto_trader.py -v            # Auto-trader tests
python -m pytest tests/test_risk_manager.py -v           # Risk manager tests
```

## Directory Structure

```
Leap/
├── main.py                    # CLI entry point with LeapTradingSystem orchestrator
├── requirements.txt           # Project dependencies
├── config/
│   └── settings.py            # Dataclass-based hierarchical configuration
├── core/
│   ├── data_pipeline.py       # Data fetching, feature engineering (100+ features)
│   ├── trading_types.py       # Shared enums and dataclasses (Action, Position, TradingState)
│   ├── trading_env_base.py    # Abstract base class for trading environments
│   ├── trading_env.py         # Gymnasium backtesting environment
│   ├── live_trading_env.py    # Live MT5 trading environment
│   ├── risk_manager.py        # Position sizing, drawdown limits, circuit breakers
│   ├── mt5_broker.py          # MetaTrader 5 gateway
│   ├── auto_trader.py         # Autonomous trading orchestrator
│   ├── order_manager.py       # Order execution and validation
│   └── position_sync.py       # Position synchronization with broker
├── models/
│   ├── transformer.py         # Temporal Fusion Transformer with quantile outputs
│   └── ppo_agent.py           # PPO actor-critic agent
├── training/
│   ├── trainer.py             # Model training orchestration
│   └── online_learning.py     # Online adaptation with regime detection
├── evaluation/
│   ├── backtester.py          # Backtesting, walk-forward, Monte Carlo
│   └── metrics.py             # 30+ performance metrics calculator
├── utils/
│   ├── device.py              # Centralized PyTorch device management
│   ├── logging_config.py      # Rotating log handler setup
│   └── mlflow_tracker.py      # MLflow experiment tracking
├── tests/                     # pytest test suite (9 files)
├── docs/
│   ├── ARCHITECTURE.md        # System architecture details
│   └── AUTO_TRADER.md         # Auto-trader documentation
├── saved_models/              # Trained model files
├── checkpoints/               # Training checkpoints
├── results/                   # Backtest results
└── logs/                      # Application logs
```

## Architecture

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| `LeapTradingSystem` | `main.py` | CLI orchestrator with lazy component initialization |
| `DataPipeline` | `core/data_pipeline.py` | Data fetching and 100+ feature engineering |
| `TransformerPredictor` | `models/transformer.py` | Price prediction with quantile uncertainty |
| `PPOAgent` | `models/ppo_agent.py` | RL agent for trading decisions |
| `RiskManager` | `core/risk_manager.py` | Position sizing and risk limits |
| `TradingEnvironment` | `core/trading_env.py` | Gymnasium backtest environment |
| `LiveTradingEnvironment` | `core/live_trading_env.py` | MT5 live/paper trading |
| `AutoTrader` | `core/auto_trader.py` | Autonomous trading orchestrator |
| `OnlineLearningManager` | `training/online_learning.py` | Continuous model adaptation |
| `MLflowTracker` | `utils/mlflow_tracker.py` | Experiment tracking |
| `Trade`, `TradeStatistics` | `core/trading_types.py` | Consolidated trade/position dataclasses |
| `get_device`, `resolve_device` | `utils/device.py` | Centralized PyTorch device utilities |

### Data Flow

```
Market Data (MT5/API)
    │
    ▼
DataPipeline → FeatureEngineer (100+ features)
    │
    ├───────────────────────────────┐
    ▼                               ▼
TransformerPredictor            PPOAgent
(price + uncertainty)           (action)
    │                               │
    └───────────┬───────────────────┘
                ▼
          RiskManager → validate trade
                │
                ▼
          Order Execution
                │
                ▼
          OnlineLearning → monitor & adapt
```

### Model Architecture

**Transformer Predictor:**
- Input projection → Positional encoding → N encoder layers
- Temporal attention → Gated residual network
- Outputs: Point prediction + Quantile outputs (0.1, 0.5, 0.9)

**PPO Agent:**
- Shared feature extractor (256→256)
- Actor head (128→4 actions) + Critic head (128→1 value)
- GAE for advantage estimation, clipped surrogate objective

### Trading Environment Hierarchy

```
BaseTradingEnvironment (ABC)
├── TradingEnvironment     # Backtesting with historical data
└── LiveTradingEnvironment # Real-time MT5 integration
```

## Configuration System

Hierarchical dataclass configuration in `config/settings.py`:

| Config Class | Purpose |
|--------------|---------|
| `SystemConfig` | Root config aggregating all sub-configs |
| `DataConfig` | Symbols, timeframes, lookback window, features |
| `TransformerConfig` | Model dimensions, layers, learning rates |
| `PPOConfig` | Actor-critic architecture, RL hyperparameters |
| `RiskConfig` | Position sizing method, drawdown limits |
| `BacktestConfig` | Initial balance, commission, walk-forward params |
| `AutoTraderConfig` | MT5 credentials, trading hours, risk per trade |
| `LoggingConfig` | Log levels, rotation, file sizing |
| `MLflowConfig` | Experiment tracking, model registry |

```python
# Load/save configuration
config = SystemConfig.load("config.json")
config.save("config.json")
```

## Key Patterns

- **Lazy Loading**: Components initialized only when needed in `LeapTradingSystem`
- **Dataclass Configuration**: Type-safe hierarchical configs with JSON serialization
- **Gymnasium Integration**: Trading environments follow Gymnasium API for RL compatibility
- **Walk-Forward Optimization**: Rolling train/test splits to prevent overfitting
- **Event-Driven Architecture**: Position synchronizer uses callbacks for state changes

## Auto-Trader System

The autonomous trading system (`core/auto_trader.py`) coordinates:

```
MT5BrokerGateway ◄──► OrderManager ◄──► PositionSynchronizer
        │                  │                      │
        └──────────────────┼──────────────────────┘
                           ▼
                     AutoTrader
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  Transformer        PPO Agent          OnlineLearning
```

**States:** `STOPPED → STARTING → RUNNING ⇄ PAUSED → STOPPING → ERROR`

**Trading Loop:**
1. Sync positions with broker
2. Check trading hours and daily limits
3. Generate signal (Transformer + PPO)
4. Validate with risk manager
5. Execute via order manager
6. Trigger online learning adaptation

## Online Learning

The `OnlineLearningManager` monitors and adapts models:

- **Market Regime Detection**: Trending, ranging, volatile states
- **Adaptation Triggers**: Prediction error threshold, drawdown, regime change
- **Catastrophic Forgetting Prevention**: Lower learning rates, validation checks

## Extension Points

| Extension | Location | Method |
|-----------|----------|--------|
| New features | `core/data_pipeline.py` | Add `FeatureEngineer._add_*()` method |
| New models | `models/` | Implement same interface, add to `__init__.py` |
| New risk methods | `core/risk_manager.py` | Add to `RiskManager.calculate_position_size()` |
| New metrics | `evaluation/metrics.py` | Add to `MetricsCalculator.calculate_all()` |
| New environments | `core/` | Extend `BaseTradingEnvironment` |
| Custom signals | `core/auto_trader.py` | Override `AutoTrader._generate_signal()` |

## Testing

**Test files in `tests/`:**
- `test_cli.py` - CLI and system integration
- `test_integration.py` - End-to-end pipeline
- `test_feature_engineering.py` - Feature computation
- `test_transformer.py` - Transformer model
- `test_ppo_agent.py` - PPO agent
- `test_trading_env.py` - Trading environments
- `test_trainer.py` - Training pipeline
- `test_risk_manager.py` - Risk management
- `test_auto_trader.py` - Auto-trader

**Testing patterns:**
- Mock classes for external dependencies (MT5, data sources)
- Fixture-based test data generation
- Integration tests for full pipelines

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Deep Learning | PyTorch >=2.2.0, NumPy <2.0 (ABI compatibility) |
| Data | Pandas >=2.0, SciPy >=1.11, scikit-learn >=1.3 |
| RL | Gymnasium >=0.29 |
| Trading | MetaTrader5 >=5.0.45 (Windows only, optional) |
| Tracking | MLflow >=2.10, TensorBoard >=2.14 |
| Configuration | dataclasses-json |
| Testing | pytest >=7.4 |

## Important Notes

- **NumPy Version**: Must be <2.0 for PyTorch ABI compatibility
- **MT5 Integration**: Only works on Windows; use paper mode on other platforms
- **Safety**: Always use `--paper` flag when testing live trading
- **Model Files**: Saved to `saved_models/` with `model_metadata.json` for reloading
- **Logging**: Logs rotate by size (10MB default) with 5 backups
- **MLflow**: Experiments tracked in `mlruns/` directory

## CLI Reference

See `CLI.md` for detailed command documentation.

| Command | Description |
|---------|-------------|
| `train` | Train Transformer + PPO models |
| `backtest` | Run backtest with optional Monte Carlo |
| `walkforward` | Walk-forward optimization |
| `live` | Start live/paper trading session |
| `evaluate` | Evaluate trained models |
| `autotrade` | Start autonomous MT5 trading |

**Global options:** `--symbol`, `--timeframe`, `--bars`, `--model-dir`, `--config`, `--log-level`, `--paper`

## Environment Variables

For live trading with MT5:
```bash
export MT5_LOGIN=12345678
export MT5_PASSWORD=your_password
export MT5_SERVER=YourBroker-Server
```

## Additional Documentation

- `docs/ARCHITECTURE.md` - Detailed system architecture with diagrams
- `docs/AUTO_TRADER.md` - Auto-trader usage and configuration
- `CLI.md` - Complete CLI command reference
