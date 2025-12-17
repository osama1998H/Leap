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
| `evaluate` | Evaluate trained models on test data |
| `autotrade` | Start auto-trader with MetaTrader5 integration (Windows only) |
| `api/run.py` | Start Web UI backend API server |
| `ui (npm run dev)` | Start Web UI frontend development server |

---

## Commands

### train

Train the Transformer price predictor and/or PPO reinforcement learning agent.

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
| `--model-type` | | choice | `both` | Model to train: `transformer`, `ppo`, or `both` |
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
# Train with default settings (uses config values, trains both models)
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

**Training Individual Models:**

Use `--model-type` to train only the Transformer predictor or only the PPO agent:

```bash
# Train only the Transformer predictor (supervised learning on price prediction)
python main.py train --model-type transformer --symbol EURUSD --epochs 100

# Train only the PPO agent (reinforcement learning on trading environment)
python main.py train --model-type ppo --symbol EURUSD --timesteps 100000

# Train both models (default behavior)
python main.py train --model-type both --symbol EURUSD --epochs 100 --timesteps 100000
```

**Use Cases for Individual Training:**
- **Transformer-only**: Fine-tune prediction model without retraining trading policy
- **PPO-only**: Retrain trading policy when market regime changes, keeping existing predictor
- **Both**: Full training pipeline (default)

---

### backtest

Run backtesting on historical market data with optional realistic trading constraints.

**Strategy Behavior:**

The backtest uses the same strategy as live trading (`autotrade`):
1. **Transformer** predicts expected returns and confidence
2. **PPO Agent** selects action based on market + account state
3. **Signal combination** validates agent decisions against predictions:
   - Agent CLOSE → always CLOSE
   - Agent BUY + prediction agrees → BUY
   - Agent BUY + prediction contradicts → HOLD (cautious)
   - Agent BUY + weak prediction → BUY (trust agent)
   - Same pattern for SELL

This ensures backtest results accurately predict live trading performance.

> **Note:** If only the Transformer is loaded (no PPO agent), the backtest runs with Transformer predictions only.

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
# Basic backtest (uses both Transformer + PPO agent if available)
python main.py backtest --symbol EURUSD

# Backtest with realistic constraints and Monte Carlo analysis
python main.py backtest --symbol EURUSD --realistic --monte-carlo

# Backtest with more historical data
python main.py backtest --bars 100000 --model-dir ./my_models
```

---

### walkforward

Run walk-forward optimization using rolling window train/test splits to validate strategy robustness.

**What Walk-Forward Does:**
1. Splits data into rolling train/test windows
2. **Trains a FRESH Transformer model** on each fold's training data
3. **Tests the trained model** on unseen future data
4. Aggregates metrics across all folds

This prevents overfitting and answers: *"Would this strategy work if deployed in real-time?"*

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
- `train_window_days`: 180 days (training window per fold)
- `test_window_days`: 30 days (testing window per fold)
- `walk_forward_epochs`: 20 (reduced training epochs per fold for speed)
- `walk_forward_parallel`: False (disabled to avoid GPU contention during training)

**Walk-Forward Process:**
```
Fold 1: Train on Jan-Jun → Test on Jul (train model A, discard after)
Fold 2: Train on Feb-Jul → Test on Aug (train model B, discard after)
Fold 3: Train on Mar-Aug → Test on Sep (train model C, discard after)
... continues rolling forward ...
```

**Output Metrics:**
- Per-fold results (return, Sharpe, drawdown, win rate, trades)
- Aggregated statistics (mean, std, min, max across all folds)
- Consistency metrics (profitable folds ratio)

**Important Notes:**
- Walk-forward trains **temporary models** for validation only
- Models are discarded after each fold - they're not saved
- After validation, use `train` command to create your production model
- Execution time depends on: number of folds × training epochs

**Examples:**
```bash
# Run walk-forward optimization (will train models, takes several minutes)
python main.py walkforward --symbol EURUSD

# Walk-forward with extended data (more folds, more robust validation)
python main.py walkforward --symbol GBPUSD --bars 100000

# Walk-forward with debug logging to see training progress
python main.py walkforward --symbol EURUSD --log-level DEBUG
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

## Web UI

The Leap Trading System includes a web-based dashboard for visual monitoring and configuration.

### Starting the Web UI

**Backend API (FastAPI):**
```bash
# Start API server on port 8000 (from project root)
python api/run.py
```

**Frontend (React + Vite):**
```bash
# Start development server on port 5173 (from project root)
cd ui && npm run dev
```

**Access the UI:** Open `http://localhost:5173` in your browser.

### Quick Start

```bash
# Terminal 1: Start backend
python api/run.py

# Terminal 2: Start frontend
cd ui && npm run dev
```

### Web UI Features

| Page | Description |
|------|-------------|
| **Dashboard** | System overview, active jobs, recent results |
| **Training** | Configure and launch training jobs |
| **Backtest** | Run backtests and view results |
| **Config** | Edit system configuration |
| **Logs** | View and search application logs |

### API Endpoints

The backend exposes REST API endpoints at `http://localhost:8000/api/v1/`:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /training/start` | Start training job |
| `GET /training/jobs` | List training jobs |
| `POST /backtest/run` | Run backtest |
| `GET /backtest/results` | List backtest results |
| `GET /config` | Get configuration |
| `PUT /config` | Update configuration |
| `GET /models` | List trained models |
| `GET /logs/files` | List log files |
| `GET /metrics/system` | System metrics (CPU, memory, GPU) |

---

## MLflow Experiment Tracking

Leap uses MLflow 3 for experiment tracking, model versioning, and metrics logging.

### Viewing Experiments

**Recommended (helper script):**
```bash
# Launch MLflow UI with proper configuration
python scripts/mlflow_ui.py --port 5000
```

**Manual method:**
```bash
# Must specify the backend-store-uri to find the database
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --port 5000
```

Open `http://localhost:5000` in your browser to view experiments.

### MLflow CLI Options

Available for `train` and `backtest` commands:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mlflow-experiment` | string | `leap-trading` | MLflow experiment name |
| `--mlflow-tracking-uri` | string | `sqlite:///mlflow.db` | MLflow tracking URI |
| `--no-mlflow` | flag | `False` | Disable MLflow tracking |

**Examples:**
```bash
# Train with custom experiment name
python main.py train --mlflow-experiment my-experiment

# Disable MLflow tracking
python main.py train --no-mlflow

# Use custom tracking server
python main.py train --mlflow-tracking-uri http://localhost:5000
```

### What Gets Tracked

| Category | Tracked Data |
|----------|--------------|
| **Parameters** | Model hyperparameters, training config, data settings |
| **Metrics** | Train/val loss, episode rewards, backtest results |
| **Artifacts** | Model weights, training history, config snapshots |
| **Datasets** | Training/validation data metadata (MLflow 3) |
| **Models** | PyTorch models with signatures (Model Registry) |

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
python main.py train --config config/example_config.json
```

### Example Configuration

See [`config/example_config.json`](config/example_config.json) for a complete example configuration file with all available options.

For detailed documentation of the configuration system (dataclasses, type safety, etc.), see [ARCHITECTURE.md](ARCHITECTURE.md#configuration-system).

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

See [README.md](README.md#project-structure) for complete project directory structure.

**CLI-specific directories:**
- `saved_models/` - Model weights, config snapshots, `model_metadata.json`
- `logs/` - Log files (`leap_YYYYMMDD_HHMMSS.log`)
- `checkpoints/` - Training checkpoints
- `results/` - Backtest and evaluation results

---

## Important Notes

### Safety
- **Always use `--paper` flag** when testing autotrade commands
- Real money trading requires explicit confirmation
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

# 5. Paper trade to validate (Windows only - requires MT5)
python main.py autotrade --paper --symbol EURUSD
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
