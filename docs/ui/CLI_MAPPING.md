# CLI to UI Feature Mapping

This document maps every CLI command, argument, and option to its corresponding UI location and component.

---

## Command Mapping

### `train` Command

| CLI Option | Short | Type | Default | UI Location | UI Component | Notes |
|------------|-------|------|---------|-------------|--------------|-------|
| `--symbol` | `-s` | string | EURUSD | Training Config → Data | `SymbolSelector` | Single symbol mode |
| `--symbols` | - | list | from config | Training Config → Data | `SymbolSelector` | Multi-select enabled |
| `--timeframe` | `-t` | string | 1h | Training Config → Data | `TimeframeSelect` | Dropdown |
| `--multi-timeframe` | - | flag | false | Training Config → Data | `Switch` | Toggle |
| `--bars` | `-b` | int | 50000 | Training Config → Data | `NumberInput` | With validation |
| `--epochs` | `-e` | int | 100 | Training Config → Transformer | `NumberInput` | Range: 1-1000 |
| `--timesteps` | - | int | 1000000 | Training Config → PPO | `NumberInput` | Range: 10k-10M |
| `--model-dir` | `-m` | string | ./saved_models | Training Config → Output | `PathInput` | Directory picker |
| `--config` | `-c` | string | - | Training Config | `ConfigLoader` | Load from template |
| `--log-level` | `-l` | choice | INFO | Training Config → Advanced | `Select` | DEBUG/INFO/WARN/ERROR |
| `--log-file` | - | string | auto | Training Config → Advanced | `PathInput` | Optional |
| `--mlflow-experiment` | - | string | leap-trading | Training Config → MLflow | `TextInput` | - |
| `--mlflow-tracking-uri` | - | string | mlruns | Training Config → MLflow | `TextInput` | - |
| `--no-mlflow` | - | flag | false | Training Config → MLflow | `Switch` | Disable tracking |

**Config-Only Options** (not in CLI, but configurable):

| Config Path | UI Location | UI Component |
|-------------|-------------|--------------|
| `transformer.d_model` | Training → Transformer | `Select` (32/64/128/256/512) |
| `transformer.n_heads` | Training → Transformer | `Select` (1/2/4/8/16) |
| `transformer.n_encoder_layers` | Training → Transformer | `NumberInput` |
| `transformer.d_ff` | Training → Transformer | `NumberInput` |
| `transformer.dropout` | Training → Transformer | `Slider` (0.0-0.5) |
| `transformer.learning_rate` | Training → Transformer | `NumberInput` (scientific) |
| `transformer.weight_decay` | Training → Transformer | `NumberInput` (scientific) |
| `transformer.batch_size` | Training → Transformer | `Select` (16/32/64/128/256) |
| `transformer.patience` | Training → Transformer | `NumberInput` |
| `ppo.learning_rate` | Training → PPO | `NumberInput` (scientific) |
| `ppo.gamma` | Training → PPO | `Slider` (0.9-0.999) |
| `ppo.gae_lambda` | Training → PPO | `Slider` (0.9-0.999) |
| `ppo.clip_epsilon` | Training → PPO | `Slider` (0.1-0.3) |
| `ppo.entropy_coef` | Training → PPO | `NumberInput` |
| `ppo.value_coef` | Training → PPO | `NumberInput` |
| `ppo.n_steps` | Training → PPO | `NumberInput` |
| `ppo.n_epochs` | Training → PPO | `NumberInput` |
| `ppo.batch_size` | Training → PPO | `Select` |
| `data.lookback_window` | Training → Data | `NumberInput` |
| `data.prediction_horizon` | Training → Data | `NumberInput` |
| `data.train_test_split` | Training → Data | `Slider` (0.5-0.9) |
| `data.validation_split` | Training → Data | `Slider` (0.05-0.2) |

---

### `backtest` Command

| CLI Option | Short | Type | Default | UI Location | UI Component | Notes |
|------------|-------|------|---------|-------------|--------------|-------|
| `--symbol` | `-s` | string | EURUSD | Backtest Config → Data | `SymbolSelector` | - |
| `--timeframe` | `-t` | string | 1h | Backtest Config → Data | `TimeframeSelect` | - |
| `--bars` | `-b` | int | 50000 | Backtest Config → Data | `NumberInput` | - |
| `--model-dir` | `-m` | string | ./saved_models | Backtest Config → Model | `ModelBrowser` | With preview |
| `--realistic` | - | flag | false | Backtest Config → Settings | `Switch` | With tooltip |
| `--monte-carlo` | - | flag | false | Backtest Config → Settings | `Switch` | Shows MC panel |
| `--config` | `-c` | string | - | Backtest Config | `ConfigLoader` | - |
| `--log-level` | `-l` | choice | INFO | Backtest Config → Advanced | `Select` | - |

**Config-Only Options**:

| Config Path | UI Location | UI Component |
|-------------|-------------|--------------|
| `backtest.initial_balance` | Backtest → Risk | `NumberInput` |
| `backtest.commission_per_lot` | Backtest → Costs | `NumberInput` |
| `backtest.slippage_pips` | Backtest → Costs | `NumberInput` |
| `backtest.spread_pips` | Backtest → Costs | `NumberInput` |
| `backtest.leverage` | Backtest → Risk | `NumberInput` |
| `backtest.n_simulations` | Backtest → Monte Carlo | `NumberInput` |
| `backtest.confidence_level` | Backtest → Monte Carlo | `Slider` |
| `risk.risk_per_trade` | Backtest → Risk | `Slider` |
| `risk.max_position_size` | Backtest → Risk | `NumberInput` |

---

### `walkforward` Command

| CLI Option | Short | Type | Default | UI Location | UI Component | Notes |
|------------|-------|------|---------|-------------|--------------|-------|
| `--symbol` | `-s` | string | EURUSD | Walk-Forward → Data | `SymbolSelector` | - |
| `--timeframe` | `-t` | string | 1h | Walk-Forward → Data | `TimeframeSelect` | - |
| `--bars` | `-b` | int | 50000 | Walk-Forward → Data | `NumberInput` | - |
| `--config` | `-c` | string | - | Walk-Forward | `ConfigLoader` | - |
| `--log-level` | `-l` | choice | INFO | Walk-Forward → Advanced | `Select` | - |

**Config-Only Options**:

| Config Path | UI Location | UI Component |
|-------------|-------------|--------------|
| `backtest.train_window_days` | Walk-Forward → Windows | `NumberInput` |
| `backtest.test_window_days` | Walk-Forward → Windows | `NumberInput` |
| `backtest.walk_forward_epochs` | Walk-Forward → Training | `NumberInput` |
| `backtest.walk_forward_parallel` | Walk-Forward → Settings | `Switch` |

---

### `evaluate` Command

| CLI Option | Short | Type | Default | UI Location | UI Component | Notes |
|------------|-------|------|---------|-------------|--------------|-------|
| `--symbol` | `-s` | string | EURUSD | Evaluate → Data | `SymbolSelector` | - |
| `--timeframe` | `-t` | string | 1h | Evaluate → Data | `TimeframeSelect` | - |
| `--bars` | `-b` | int | 50000 | Evaluate → Data | `NumberInput` | - |
| `--model-dir` | `-m` | string | ./saved_models | Evaluate → Model | `ModelBrowser` | - |

---

### `autotrade` Command

| CLI Option | Short | Type | Default | UI Location | UI Component | Notes |
|------------|-------|------|---------|-------------|--------------|-------|
| `--symbol` | `-s` | string | EURUSD | Auto-Trader → Config | `SymbolSelector` | - |
| `--timeframe` | `-t` | string | 1h | Auto-Trader → Config | `TimeframeSelect` | - |
| `--model-dir` | `-m` | string | ./saved_models | Auto-Trader → Model | `ModelBrowser` | - |
| `--paper` | - | flag | false | Auto-Trader → Mode | `Switch` | **Always default ON** |
| `--config` | `-c` | string | - | Auto-Trader | `ConfigLoader` | - |

**Config-Only Options**:

| Config Path | UI Location | UI Component |
|-------------|-------------|--------------|
| `auto_trader.risk_per_trade` | Auto-Trader → Risk | `Slider` |
| `auto_trader.max_positions` | Auto-Trader → Risk | `NumberInput` |
| `auto_trader.default_sl_pips` | Auto-Trader → Risk | `NumberInput` |
| `auto_trader.default_tp_pips` | Auto-Trader → Risk | `NumberInput` |
| `auto_trader.trading_start_hour` | Auto-Trader → Schedule | `TimePicker` |
| `auto_trader.trading_end_hour` | Auto-Trader → Schedule | `TimePicker` |
| `auto_trader.min_confidence` | Auto-Trader → Signals | `Slider` |
| `auto_trader.enable_online_learning` | Auto-Trader → Learning | `Switch` |

---

## Complex Workflows

### Workflow: Multi-Symbol Training

**CLI Approach**:
```bash
# Option 1: Native multi-symbol (one command)
python main.py train --symbols EURUSD GBPUSD USDJPY --epochs 100

# Option 2: Loop (for different configs per symbol)
for symbol in EURUSD GBPUSD USDJPY; do
    python main.py train --symbol $symbol --model-dir ./models_$symbol
done
```

**UI Approach**:
1. Training Config → Data → Symbol selector supports multi-select
2. Toggle "Train All Symbols Together" vs "Queue Individual Jobs"
3. If together: Single job trains all symbols sequentially
4. If queue: Creates separate jobs with progress tracking for each

**UI Advantages**:
- Visual progress for each symbol
- Parallel job execution option
- Easy result comparison across symbols

---

### Workflow: Hyperparameter Tuning

**CLI Approach**:
```bash
# Manual iteration
python main.py train --epochs 50 --config config/lr_1e4.json
python main.py train --epochs 50 --config config/lr_5e5.json
python main.py train --epochs 50 --config config/lr_1e5.json
# Compare results in MLflow UI
```

**UI Approach**:
1. Training Config page with "Clone Run" button
2. Modify parameters for each variant
3. Launch all as a "Sweep" with visual comparison
4. Side-by-side loss curves in real-time
5. Automatic best run highlighting

**UI Advantages**:
- No manual config file management
- Real-time comparison
- Automatic sweep organization

---

### Workflow: Strategy Validation Pipeline

**CLI Approach**:
```bash
# Step 1: Train
python main.py train --symbol EURUSD --epochs 100 --timesteps 100000

# Step 2: Basic backtest
python main.py backtest --symbol EURUSD

# Step 3: Realistic backtest
python main.py backtest --symbol EURUSD --realistic --monte-carlo

# Step 4: Walk-forward validation
python main.py walkforward --symbol EURUSD --bars 100000

# Step 5: Evaluate on fresh data
python main.py evaluate --symbol EURUSD
```

**UI Approach**:
1. Dashboard "New Strategy Pipeline" wizard
2. Configure training parameters (Step 1)
3. Auto-queue backtests after training completes (Step 2-3)
4. Walk-forward runs with training integrated (Step 4)
5. Single results page showing full pipeline

**UI Advantages**:
- Automated pipeline execution
- Single view for all validation steps
- Clear pass/fail criteria at each stage

---

### Workflow: Model Comparison

**CLI Approach**:
```bash
# Backtest with model A
python main.py backtest --model-dir ./models_v1 > results_v1.json

# Backtest with model B
python main.py backtest --model-dir ./models_v2 > results_v2.json

# Manual comparison in spreadsheet or Python
```

**UI Approach**:
1. Backtest Results page with multi-select
2. "Compare Selected" button
3. Side-by-side metrics table
4. Overlaid equity curves on same chart
5. Statistical significance indicators

**UI Advantages**:
- Visual comparison with aligned charts
- One-click export of comparison report
- Statistical significance testing built-in

---

### Workflow: Configuration Management

**CLI Approach**:
```bash
# Copy and modify JSON files manually
cp config/base.json config/aggressive.json
vim config/aggressive.json  # Manual editing

# Use config
python main.py train --config config/aggressive.json
```

**UI Approach**:
1. Config Editor page with visual form
2. "Save as Template" button
3. Template library with descriptions
4. "Diff View" to compare templates
5. Import/Export JSON

**UI Advantages**:
- No JSON syntax errors
- Built-in validation
- Visual diff between configs
- Searchable template library

---

## Feature Parity Checklist

### Full Parity (UI matches CLI completely)

- [ ] `train` command - all options
- [ ] `backtest` command - all options
- [ ] `walkforward` command - all options
- [ ] `evaluate` command - all options
- [ ] Configuration loading/saving
- [ ] Log level configuration
- [ ] MLflow experiment configuration

### Enhanced in UI (better than CLI)

- [ ] Real-time training monitoring (CLI has no built-in monitoring)
- [ ] Visual backtest results (CLI outputs JSON)
- [ ] Side-by-side run comparison (CLI requires manual comparison)
- [ ] Pipeline workflows (CLI requires multiple commands)
- [ ] Configuration validation (CLI fails at runtime)

### Partial Parity (subset of CLI features)

- [ ] `autotrade` command - paper mode only (live trading too risky for UI)
- [ ] Advanced PPO hyperparameters (hidden in "Advanced" section)

### Not in UI (CLI only)

- [ ] Direct file path specification (UI uses directory browser)
- [ ] Shell scripting integration (UI is interactive)
- [ ] Batch processing via command line (UI uses job queue)

---

## UI Simplifications

### Smart Defaults

| Setting | CLI Default | UI Default | Rationale |
|---------|-------------|------------|-----------|
| Epochs | 100 | 100 | Good starting point |
| Timesteps | 1000000 | 100000 | Faster for initial testing |
| Realistic Mode | false | true | Safer, more accurate results |
| Monte Carlo | false | true | Valuable risk insights |
| Paper Mode (autotrade) | false | **true (locked)** | Safety first |

### Hidden Complexity

These options are available in "Advanced Settings" accordion:

1. All logging configuration
2. Device selection (auto works well)
3. Seed configuration
4. Mixed precision settings
5. Number of data loader workers
6. Detailed PPO hyperparameters
7. Online learning configuration

### Presets

| Preset Name | Description | Key Settings |
|-------------|-------------|--------------|
| Quick Test | Fast iteration | epochs=20, timesteps=50000, bars=10000 |
| Standard | Balanced training | epochs=100, timesteps=500000, bars=50000 |
| Production | Full training | epochs=200, timesteps=1000000, bars=100000 |
| Conservative Risk | Lower risk | risk_per_trade=0.01, max_positions=2 |
| Aggressive | Higher risk | risk_per_trade=0.03, max_positions=5 |
