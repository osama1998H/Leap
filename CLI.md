# Leap Trading System - CLI Reference

This document provides comprehensive documentation for all CLI commands available in the Leap trading system.

## Overview

Leap uses a command-based CLI pattern powered by Python's `argparse`. All commands are executed through `main.py`:

```bash
python main.py [COMMAND] [OPTIONS]
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `train` | Train Transformer predictor and PPO reinforcement learning agent |
| `backtest` | Run backtest on historical market data |
| `walkforward` | Run walk-forward optimization with rolling train/test splits |
| `live` | Start live or paper trading session |
| `evaluate` | Evaluate trained models on test data |
| `autotrade` | Start auto-trader with MetaTrader5 integration (Windows only) |

---

## Commands

### train

Train both the Transformer price predictor and PPO reinforcement learning agent.

**Syntax:**
```bash
python main.py train [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--symbol` | `-s` | string | from config | Trading symbol (e.g., EURUSD, GBPUSD, USDJPY) |
| `--symbols` | | list | from config | Multiple symbols for training (e.g., `--symbols EURUSD GBPUSD`) |
| `--timeframe` | `-t` | string | from config | Primary timeframe for data (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w) |
| `--multi-timeframe` | | flag | `False` | Enable multi-timeframe features (uses `additional_timeframes` from config) |
| `--bars` | `-b` | integer | `50000` | Number of historical bars to load for training |
| `--epochs` | `-e` | integer | from config | Training epochs for the Transformer predictor |
| `--timesteps` | | integer | from config | Training timesteps for the PPO agent |
| `--model-dir` | `-m` | string | `./saved_models` | Directory to save trained models |
| `--config` | `-c` | string | | Path to configuration JSON file |
| `--log-level` | `-l` | choice | | Logging level: DEBUG, INFO, WARNING, ERROR |
| `--log-file` | | string | | Custom log file path |

**Multi-Symbol Training:**
When using `--symbols` or multiple symbols in config, models are trained for each symbol:
- Models saved to `{model-dir}/{symbol}/` for each symbol
- MLflow tracks each symbol's training separately

**Multi-Timeframe Features:**
When `--multi-timeframe` is enabled:
- Fetches data for additional timeframes defined in config (`additional_timeframes`)
- Key indicators from higher/lower timeframes are added as features
- Provides multi-scale market context (e.g., 1h training with 15m/4h/1d features)

**Training Flow:**
1. Loads market data (OHLCV + 100+ computed features)
2. Prepares training sequences (lookback_window=120, prediction_horizon=12)
3. Trains Transformer predictor (supervised learning on price prediction)
4. Trains PPO agent (reinforcement learning on trading environment)
5. Saves models and metadata to specified directory

**Examples:**
```bash
# Train with default settings (uses config values)
python main.py train

# Train on GBPUSD with custom epochs
python main.py train --symbol GBPUSD --epochs 50 --timesteps 50000

# Train with a configuration file
python main.py train --config config/custom_config.json

# Train with verbose logging
python main.py train --log-level DEBUG --log-file ./training.log

# Multi-symbol training (from config or CLI)
python main.py train --symbols EURUSD GBPUSD USDJPY

# Multi-timeframe training (uses additional_timeframes from config)
python main.py train --multi-timeframe --config config/my_config.json

# Full multi-symbol + multi-timeframe training
python main.py train --symbols EURUSD GBPUSD --multi-timeframe
```

---

### backtest

Run backtesting on historical market data with optional realistic trading constraints.

**Syntax:**
```bash
python main.py backtest [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--symbol` | `-s` | string | `EURUSD` | Trading symbol |
| `--timeframe` | `-t` | string | `1h` | Timeframe for data |
| `--bars` | `-b` | integer | `50000` | Number of historical bars to load |
| `--model-dir` | `-m` | string | `./saved_models` | Directory to load trained models from |
| `--realistic` | | flag | `False` | Enable realistic trading constraints |
| `--monte-carlo` | | flag | `False` | Run Monte Carlo simulation for risk analysis |
| `--config` | `-c` | string | | Path to configuration JSON file |
| `--log-level` | `-l` | choice | | Logging level: DEBUG, INFO, WARNING, ERROR |
| `--log-file` | | string | | Custom log file path |

**Realistic Mode Constraints:**
When `--realistic` is enabled, the following constraints are applied:
- Minimum 4 hours between trades
- Maximum 5 trades per day
- Maximum 10 lots position size

**Output Metrics:**
- Total return (%)
- Sharpe ratio
- Maximum drawdown (%)
- Win rate (%)
- Total number of trades
- Monte Carlo risk analysis (if enabled)

**Examples:**
```bash
# Basic backtest
python main.py backtest --symbol EURUSD

# Backtest with realistic constraints and Monte Carlo analysis
python main.py backtest --symbol EURUSD --realistic --monte-carlo

# Backtest with more historical data
python main.py backtest --bars 100000 --model-dir ./my_models
```

---

### walkforward

Run walk-forward optimization using rolling window train/test splits to prevent overfitting.

**Syntax:**
```bash
python main.py walkforward [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--symbol` | `-s` | string | `EURUSD` | Trading symbol |
| `--timeframe` | `-t` | string | `1h` | Timeframe for data |
| `--bars` | `-b` | integer | `50000` | Number of historical bars to load |
| `--config` | `-c` | string | | Path to configuration JSON file |
| `--log-level` | `-l` | choice | | Logging level: DEBUG, INFO, WARNING, ERROR |
| `--log-file` | | string | | Custom log file path |

**Walk-Forward Parameters** (configured via config file):
- `train_window_days`: 180 days (training window)
- `test_window_days`: 30 days (testing window)
- `walk_forward_steps`: 12 (number of rolling steps)

**Examples:**
```bash
# Run walk-forward optimization
python main.py walkforward --symbol EURUSD

# Walk-forward with extended data
python main.py walkforward --symbol GBPUSD --bars 100000
```

---

### live

Start a live or paper trading session.

**Syntax:**
```bash
python main.py live [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--symbol` | `-s` | string | `EURUSD` | Trading symbol |
| `--timeframe` | `-t` | string | `1h` | Timeframe for data |
| `--model-dir` | `-m` | string | `./saved_models` | Directory to load trained models from |
| `--paper` | | flag | `False` | Enable paper trading mode (simulated trades) |
| `--config` | `-c` | string | | Path to configuration JSON file |
| `--log-level` | `-l` | choice | | Logging level: DEBUG, INFO, WARNING, ERROR |
| `--log-file` | | string | | Custom log file path |

> **Warning:** Always use `--paper` flag when testing to avoid real money trades.

**Examples:**
```bash
# Paper trading (recommended for testing)
python main.py live --paper

# Paper trading with specific symbol
python main.py live --paper --symbol GBPUSD --timeframe 4h
```

---

### evaluate

Evaluate trained models on test data and report performance metrics.

**Syntax:**
```bash
python main.py evaluate [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--symbol` | `-s` | string | `EURUSD` | Trading symbol |
| `--timeframe` | `-t` | string | `1h` | Timeframe for data |
| `--bars` | `-b` | integer | `50000` | Number of historical bars to load |
| `--model-dir` | `-m` | string | `./saved_models` | Directory to load trained models from |
| `--config` | `-c` | string | | Path to configuration JSON file |
| `--log-level` | `-l` | choice | | Logging level: DEBUG, INFO, WARNING, ERROR |
| `--log-file` | | string | | Custom log file path |

**Evaluation Metrics:**
- Predictor MAE (Mean Absolute Error) - measures prediction accuracy
- Agent Average Reward - measures RL agent performance on trading environment

**Examples:**
```bash
# Evaluate models
python main.py evaluate --model-dir ./saved_models

# Evaluate with specific symbol and data size
python main.py evaluate --symbol EURUSD --bars 10000
```

---

### autotrade

Start the auto-trader with MetaTrader5 integration for automated trading.

**Syntax:**
```bash
python main.py autotrade [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--symbol` | `-s` | string | `EURUSD` | Trading symbol |
| `--timeframe` | `-t` | string | `1h` | Timeframe for data |
| `--model-dir` | `-m` | string | `./saved_models` | Directory to load trained models from |
| `--paper` | | flag | `False` | Enable paper trading mode |
| `--config` | `-c` | string | | Path to configuration JSON file |
| `--log-level` | `-l` | choice | | Logging level: DEBUG, INFO, WARNING, ERROR |
| `--log-file` | | string | | Custom log file path |

**Requirements:**
- **Windows only** - MetaTrader5 Python package only works on Windows
- **Environment variables** for MT5 credentials:
  - `MT5_LOGIN` - Your MT5 account login
  - `MT5_PASSWORD` - Your MT5 account password
  - `MT5_SERVER` - Your MT5 broker server

**Safety:**
- Live trading requires typing `CONFIRM` when prompted
- Always test with `--paper` mode first

**Session Output:**
- Total trades executed
- Win rate
- Profit/Loss (P&L)
- Maximum drawdown

**Examples:**
```bash
# Paper trading with MT5 (safe testing)
python main.py autotrade --paper --symbol EURUSD

# Set credentials and run
export MT5_LOGIN=12345678
export MT5_PASSWORD=your_password
export MT5_SERVER=YourBroker-Server
python main.py autotrade --paper
```

---

## Global Options

These options are available for all commands:

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--config` | `-c` | string | Path to JSON configuration file (overrides defaults) |
| `--log-level` | `-l` | choice | Logging verbosity: DEBUG, INFO, WARNING, ERROR |
| `--log-file` | | string | Custom path for log file output |

**Priority:** CLI arguments override configuration file values.

---

## Configuration File

You can use a JSON configuration file to customize system behavior. Pass it with `--config`:

```bash
python main.py train --config config/my_config.json
```

### Example Configuration

```json
{
  "base_dir": "/home/user/Leap",
  "models_dir": "saved_models",
  "logs_dir": "logs",
  "device": "auto",
  "seed": 42,

  "data": {
    "symbols": ["EURUSD", "GBPUSD"],
    "primary_timeframe": "1h",
    "additional_timeframes": ["15m", "4h", "1d"],
    "lookback_window": 120,
    "prediction_horizon": 12,
    "train_test_split": 0.8,
    "validation_split": 0.1,
    "use_technical_indicators": true,
    "use_price_patterns": true,
    "use_volume_features": true,
    "normalize_method": "robust"
  },

  "transformer": {
    "d_model": 128,
    "n_heads": 8,
    "n_encoder_layers": 4,
    "d_ff": 512,
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "epochs": 100,
    "patience": 15
  },

  "ppo": {
    "actor_hidden_sizes": [256, 256, 128],
    "critic_hidden_sizes": [256, 256, 128],
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "n_steps": 2048,
    "n_epochs": 10,
    "batch_size": 64,
    "total_timesteps": 1000000
  },

  "risk": {
    "max_position_size": 0.02,
    "max_daily_loss": 0.05,
    "max_drawdown": 0.15,
    "default_stop_loss_pips": 50,
    "default_take_profit_pips": 100,
    "risk_reward_ratio": 2.0,
    "max_open_positions": 5,
    "use_trailing_stop": true,
    "trailing_stop_pips": 30
  },

  "backtest": {
    "initial_balance": 10000.0,
    "commission_per_lot": 7.0,
    "slippage_pips": 1.0,
    "spread_pips": 1.5,
    "leverage": 100,
    "train_window_days": 180,
    "test_window_days": 30,
    "walk_forward_steps": 12,
    "n_simulations": 1000,
    "confidence_level": 0.95
  },

  "logging": {
    "level": "INFO",
    "log_to_file": true,
    "log_to_console": true,
    "max_file_size_mb": 10,
    "backup_count": 5
  },

  "auto_trader": {
    "symbols": ["EURUSD"],
    "timeframe": "1h",
    "risk_per_trade": 0.01,
    "max_positions": 3,
    "default_sl_pips": 50.0,
    "default_tp_pips": 100.0,
    "trading_start_hour": 0,
    "trading_end_hour": 24,
    "paper_mode": true,
    "min_confidence": 0.6,
    "enable_online_learning": true,
    "magic_number": 234567
  }
}
```

### Configuration Sections

| Section | Description |
|---------|-------------|
| `data` | Data loading, feature engineering, and preprocessing settings |
| `transformer` | Transformer model architecture and training hyperparameters |
| `ppo` | PPO reinforcement learning agent configuration |
| `risk` | Risk management parameters (position sizing, stop-loss, etc.) |
| `backtest` | Backtesting simulation settings |
| `logging` | Log output configuration |
| `auto_trader` | Auto-trading specific settings |

---

## Directory Structure

The CLI creates and uses the following directory structure:

```
Leap/
├── saved_models/              # Trained model files (default)
│   ├── predictor.pt          # Transformer model weights
│   ├── agent.pt              # PPO agent weights
│   ├── model_metadata.json   # Model dimensions for loading
│   └── config.json           # Configuration snapshot
│
├── logs/                      # Log files
│   └── leap_YYYYMMDD_HHMMSS.log
│
├── checkpoints/               # Training checkpoints
│   └── checkpoint_epoch_XX.pt
│
├── results/                   # Backtest and evaluation results
│   └── backtest_YYYYMMDD_HHMMSS.json
│
└── data/                      # Market data cache
```

---

## Important Notes

### Safety
- **Always use `--paper` flag** when testing live or autotrade commands
- Live trading with real money requires explicit confirmation
- Never share or commit MT5 credentials

### Platform Requirements
- **MetaTrader5 integration** (`autotrade` command) only works on **Windows**
- NumPy must be < 2.0 for PyTorch compatibility

### Model Loading
- Models are automatically saved with metadata (`model_metadata.json`)
- This metadata is required for proper model reloading
- Always use the same `--model-dir` for training and inference

### Logging
- CLI `--log-level` overrides config file settings
- Log files are automatically rotated based on size or time
- Use `DEBUG` level for troubleshooting

### Performance Tips
- Start with smaller `--bars` values for initial testing
- Use `--realistic` mode for production-quality backtests
- Enable `--monte-carlo` for risk analysis on important backtests

---

## Examples

### Complete Training Workflow

```bash
# 1. Train models
python main.py train --symbol EURUSD --timeframe 1h --epochs 100 --timesteps 100000

# 2. Evaluate trained models
python main.py evaluate --model-dir ./saved_models

# 3. Run backtest with realistic constraints
python main.py backtest --symbol EURUSD --realistic --monte-carlo

# 4. Run walk-forward optimization
python main.py walkforward --symbol EURUSD --bars 100000

# 5. Paper trade to validate
python main.py live --paper --symbol EURUSD
```

### Using Custom Configuration

```bash
# Create a config file, then use it across commands
python main.py train --config config/aggressive.json
python main.py backtest --config config/aggressive.json --realistic
python main.py evaluate --config config/aggressive.json
```

### Multi-Symbol Analysis

```bash
# Method 1: Native multi-symbol training (recommended)
# Trains on all symbols from config or CLI in one command
python main.py train --symbols EURUSD GBPUSD USDJPY

# Method 2: Shell loop for more control
for symbol in EURUSD GBPUSD USDJPY; do
    python main.py train --symbol $symbol --model-dir ./models_$symbol
    python main.py backtest --symbol $symbol --model-dir ./models_$symbol --realistic
done
```

### Multi-Timeframe Training

```bash
# Enable multi-timeframe features (uses additional_timeframes from config)
# Config example: "additional_timeframes": ["15m", "4h", "1d"]
python main.py train --multi-timeframe --config config/my_config.json

# Multi-symbol + multi-timeframe combined
python main.py train --symbols EURUSD GBPUSD --multi-timeframe --config config/my_config.json
```
