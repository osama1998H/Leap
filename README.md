# Leap Trading System

An advanced AI-powered forex trading system with online learning and adaptation capabilities. Combines Transformer-based price prediction with PPO reinforcement learning for intelligent trading decisions.

## Features

- **Transformer-based Price Prediction**: Temporal Fusion Transformer for accurate time-series forecasting with uncertainty estimation
- **PPO Reinforcement Learning**: Proximal Policy Optimization agent for optimal trading decisions
- **Online Learning**: Continuous model adaptation to changing market conditions
- **Market Regime Detection**: Automatic detection of market states (trending, ranging, high volatility)
- **Comprehensive Backtesting**: Walk-forward optimization and Monte Carlo simulation
- **Risk Management**: Dynamic position sizing, stop-loss/take-profit, and drawdown limits
- **Multi-timeframe Analysis**: Feature engineering across multiple timeframes

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LEAP TRADING SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Market     │───▶│    Data      │───▶│   Feature    │                   │
│  │    Data      │    │   Pipeline   │    │  Engineering │                   │
│  │  (MT5/API)   │    │              │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                  │                           │
│                      ┌───────────────────────────┼───────────────────────┐  │
│                      │                           ▼                       │  │
│                      │  ┌─────────────────────────────────────────────┐ │  │
│                      │  │              AI MODELS LAYER                 │ │  │
│                      │  ├─────────────────────┬───────────────────────┤ │  │
│                      │  │                     │                       │ │  │
│                      │  │  ┌───────────────┐  │  ┌─────────────────┐  │ │  │
│                      │  │  │  Transformer  │  │  │   PPO Agent     │  │ │  │
│                      │  │  │  Predictor    │  │  │  (RL Trading)   │  │ │  │
│                      │  │  │               │  │  │                 │  │ │  │
│                      │  │  │ - Attention   │  │  │ - Actor-Critic  │  │ │  │
│                      │  │  │ - Quantiles   │  │  │ - GAE           │  │ │  │
│                      │  │  │ - Uncertainty │  │  │ - Clipped Loss  │  │ │  │
│                      │  │  └───────┬───────┘  │  └────────┬────────┘  │ │  │
│                      │  │          │          │           │           │ │  │
│                      │  └──────────┼──────────┴───────────┼───────────┘ │  │
│                      │             │                      │             │  │
│                      │             ▼                      ▼             │  │
│                      │  ┌──────────────────────────────────────────┐   │  │
│                      │  │         ONLINE LEARNING MANAGER          │   │  │
│                      │  │                                          │   │  │
│                      │  │  - Performance Monitoring                │   │  │
│                      │  │  - Regime Detection                      │   │  │
│                      │  │  - Adaptive Model Updates                │   │  │
│                      │  │  - Catastrophic Forgetting Prevention    │   │  │
│                      │  └──────────────────────────────────────────┘   │  │
│                      └─────────────────────────────────────────────────┘  │
│                                          │                                 │
│                                          ▼                                 │
│                      ┌─────────────────────────────────────────────────┐  │
│                      │              RISK MANAGEMENT                     │  │
│                      │                                                  │  │
│                      │  - Position Sizing (Kelly/Volatility/Fixed)     │  │
│                      │  - Stop Loss / Take Profit                      │  │
│                      │  - Drawdown Limits                              │  │
│                      │  - Daily/Weekly Loss Limits                     │  │
│                      └─────────────────────────────────────────────────┘  │
│                                          │                                 │
│                                          ▼                                 │
│                      ┌─────────────────────────────────────────────────┐  │
│                      │              EXECUTION LAYER                     │  │
│                      │                                                  │  │
│                      │  - Backtest Engine                              │  │
│                      │  - Paper Trading                                │  │
│                      │  - Live Trading (MT5)                           │  │
│                      └─────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Architecture

For detailed technical architecture documentation including:
- **Model Architecture**: Transformer Predictor and PPO Agent internals
- **Data Flow**: Complete data pipeline and feature engineering details
- **Online Learning**: Adaptation triggers and learning loops
- **Walk-Forward Optimization**: Backtesting methodology

See [ARCHITECTURE.md](ARCHITECTURE.md) for comprehensive diagrams and technical details.

## Project Structure

```
Leap/
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration management (all dataclasses)
│
├── core/
│   ├── __init__.py
│   ├── data_pipeline.py         # Data fetching & feature engineering
│   ├── trading_env.py           # Gymnasium trading environment (backtest)
│   ├── trading_env_base.py      # Abstract base trading environment
│   ├── live_trading_env.py      # Live trading environment (MT5)
│   ├── trading_types.py         # Shared trading types and dataclasses
│   ├── risk_manager.py          # Position sizing & risk limits
│   ├── mt5_broker.py            # MetaTrader 5 broker gateway
│   ├── order_manager.py         # Order execution and validation
│   ├── position_sync.py         # Position synchronization with broker
│   └── auto_trader.py           # Autonomous trading orchestrator
│
├── models/
│   ├── __init__.py
│   ├── transformer.py           # Temporal Fusion Transformer
│   └── ppo_agent.py             # PPO reinforcement learning
│
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Model training orchestration
│   └── online_learning.py       # Online adaptation system
│
├── evaluation/
│   ├── __init__.py
│   ├── backtester.py            # Backtesting engine
│   └── metrics.py               # Performance metrics
│
├── utils/
│   ├── __init__.py
│   └── logging_config.py        # Unified logging configuration
│
├── tests/
│   ├── __init__.py
│   ├── test_integration.py      # Integration tests
│   ├── test_auto_trader.py      # Auto-trader tests
│   └── test_feature_engineering.py  # Feature engineering tests
│
├── docs/
│   └── AUTO_TRADER.md           # Auto-trader documentation
│
├── ARCHITECTURE.md              # System architecture documentation
│
├── main.py                      # CLI entry point
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Leap.git
cd Leap

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For MetaTrader 5 integration (Windows only)
pip install MetaTrader5
```

## Usage

### Training

```bash
# Train both prediction and RL models
python main.py train --symbol EURUSD --timeframe 1h --epochs 100 --timesteps 100000

# Train with custom configuration
python main.py train --config config/custom_config.json

# Multi-symbol training (trains each symbol sequentially)
python main.py train --symbols EURUSD GBPUSD USDJPY

# Multi-timeframe training (uses additional_timeframes from config)
python main.py train --multi-timeframe --config config/my_config.json

# Multi-symbol + multi-timeframe combined
python main.py train --symbols EURUSD GBPUSD --multi-timeframe
```

> **Note:** Multi-timeframe training adds indicators from higher/lower timeframes as features (e.g., 1h training with 15m/4h/1d features). See [CLI.md](CLI.md) for detailed options.

### Backtesting

```bash
# Run backtest on historical data
python main.py backtest --symbol EURUSD --bars 50000

# Backtest with realistic constraints (limited trades, capped position size)
python main.py backtest --symbol EURUSD --realistic

# Backtest with Monte Carlo risk analysis
python main.py backtest --symbol EURUSD --monte-carlo

# Walk-forward optimization
python main.py walkforward --symbol EURUSD
```

### Evaluation

```bash
# Evaluate trained models
python main.py evaluate --model-dir ./saved_models
```

### Live/Auto-Trading

The auto-trader runs autonomously, combining Transformer predictions with PPO agent decisions.
See [docs/AUTO_TRADER.md](docs/AUTO_TRADER.md) for detailed documentation.

> **Note:** Live trading requires Windows with MetaTrader 5 integration.

```bash
# Paper trading mode (recommended for testing)
python main.py autotrade --paper --symbol EURUSD

# Live auto-trading (requires confirmation, Windows + MT5 only)
python main.py autotrade --symbol EURUSD

# With specific model directory
python main.py autotrade --paper --model-dir ./saved_models --symbol GBPUSD
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run integration tests
python -m pytest tests/test_integration.py -v

# Run auto-trader tests
python -m pytest tests/test_auto_trader.py -v

# Run feature engineering tests
python -m pytest tests/test_feature_engineering.py -v
```

### CLI Options

Common options available for most commands:

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--symbol` | `-s` | EURUSD | Trading symbol |
| `--symbols` | | | Multiple symbols (e.g., `--symbols EURUSD GBPUSD`) |
| `--timeframe` | `-t` | 1h | Timeframe |
| `--multi-timeframe` | | False | Enable multi-timeframe features |
| `--config` | `-c` | | Path to config file |
| `--paper` | | False | Use paper trading mode |

> **Complete Reference:** See [CLI.md](CLI.md) for full command documentation including all options, examples, and configuration details.

## Configuration

All configuration is managed via dataclasses in `config/settings.py`. The main `SystemConfig` contains nested configuration objects.

**Quick Start:**
```bash
# Use example config
python main.py train --config config/example_config.json
```

**Key Configuration Sections:**
- `data` - Symbols, timeframes, feature engineering
- `transformer` - Model architecture, learning rate
- `ppo` - RL hyperparameters (gamma, clip_epsilon, etc.)
- `risk` - Position sizing, drawdown limits
- `auto_trader` - Trading hours, risk per trade

> **Full Documentation:** See [ARCHITECTURE.md](ARCHITECTURE.md#configuration-system) for complete configuration reference and [`config/example_config.json`](config/example_config.json) for all available options.

## Performance Metrics

The system calculates comprehensive metrics:

- **Return Metrics**: Total return, CAGR, annualized return
- **Risk Metrics**: Volatility, max drawdown, VaR, CVaR
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trade Statistics**: Win rate, profit factor, payoff ratio
- **Distribution**: Skewness, kurtosis, tail ratio

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading forex involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always use paper trading to test strategies before risking real capital.
