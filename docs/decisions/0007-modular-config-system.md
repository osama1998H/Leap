# ADR 0007: Modular Configuration System

## Status
Accepted

## Context

The existing `--config` flag loads a monolithic `SystemConfig` JSON file that contains all configuration sections (data, transformer, ppo, risk, backtest, logging, auto_trader, etc.). This creates several issues:

1. **Config file size**: Users must maintain large config files even for simple customizations
2. **Command-specific needs**: Different commands need different config sections (train vs backtest vs autotrade)
3. **Manageability**: Large config files become unwieldy and hard to version control
4. **Separation of concerns**: Training hyperparameters shouldn't be mixed with backtest parameters

Users frequently want to change only training parameters or only backtest settings without modifying a 100+ line config file.

## Decision

Replace the monolithic `--config` flag with modular config flags, each loading a standalone config file for a specific purpose:

| Flag | Config Type | Used By |
|------|-------------|---------|
| `--training-config` | TransformerConfig + PPOConfig + device/seed | `train` |
| `--data-config` | DataConfig | `train`, `backtest`, `walkforward` |
| `--backtest-config` | BacktestConfig | `backtest`, `walkforward` |
| `--risk-config` | RiskConfig | All commands |
| `--auto-trader-config` | AutoTraderConfig | `autotrade` |
| `--logging-config` | LoggingConfig | All commands |

Each config file is standalone and self-contained. The system starts with default values from `get_config()` and overrides only the sections specified by the provided config files.

### Implementation

1. **Standalone loader functions** in `config/settings.py`:
   - `load_training_config(path)` - Returns (TransformerConfig, PPOConfig, device, seed)
   - `load_data_config(path)` - Returns DataConfig
   - `load_backtest_config(path)` - Returns BacktestConfig
   - `load_risk_config(path)` - Returns RiskConfig
   - `load_auto_trader_config(path)` - Returns AutoTraderConfig
   - `load_logging_config(path)` - Returns LoggingConfig

2. **CLI argument parsing** in `main.py`:
   - Remove `--config` flag
   - Add new modular config flags
   - Load defaults, then override with any provided config files

3. **Template config files** in `config/templates/`:
   - `training.json`, `data.json`, `backtest.json`, `risk.json`, `auto_trader.json`, `logging.json`

### Example Usage

```bash
# Training with custom hyperparameters
python main.py train --training-config config/my_training.json --symbol EURUSD

# Backtesting with custom simulation parameters
python main.py backtest --backtest-config config/my_backtest.json --symbol EURUSD

# Combining multiple configs
python main.py train \
    --training-config config/training.json \
    --data-config config/data.json \
    --logging-config config/logging.json
```

### Example `training.json`
```json
{
  "device": "auto",
  "seed": 42,
  "transformer": {
    "d_model": 128,
    "n_heads": 8,
    "learning_rate": 1e-4,
    "epochs": 100
  },
  "ppo": {
    "learning_rate": 3e-4,
    "total_timesteps": 1000000
  }
}
```

## Consequences

### Positive
- **Smaller, focused config files**: Users only need to specify what they want to change
- **Easier version control**: Small config files are easier to diff and review
- **Better separation of concerns**: Training, backtesting, and live trading configs are separate
- **Flexibility**: Users can combine configs as needed for different use cases
- **Backwards compatible defaults**: Running without any config files still works with defaults

### Negative
- **Breaking change**: The `--config` flag is removed entirely
- **More CLI flags**: Users must learn multiple new flags
- **Multiple files**: Some users may prefer a single config file

### Migration

Users with existing monolithic config files must split them into separate files:

1. Extract `transformer` and `ppo` sections into `training.json`
2. Extract `data` section into `data.json`
3. Extract `backtest` section into `backtest.json`
4. Extract `risk` section into `risk.json`
5. Extract `auto_trader` section into `auto_trader.json`
6. Extract `logging` section into `logging.json`

Template files are provided in `config/templates/` as starting points.

## Related Decisions

- [ADR-0003: Dataclass-based Configuration](0003-dataclass-configuration.md) - Established the dataclass pattern
- [ADR-0001: Centralized Utilities](0001-centralized-utilities.md) - Loader functions follow centralized pattern
