---
paths: cli/**/*.py
---
# CLI Module Development Rules

These rules apply when working on files in the `cli/` directory.

---

## Module Overview

The `cli/` module handles command-line interface:

| File | Responsibility |
|------|----------------|
| `__init__.py` | Main entry point `main()`, exports |
| `parser.py` | Argument parsing, CLI structure |
| `system.py` | `LeapTradingSystem` orchestrator |
| `commands/__init__.py` | Command registry, `execute_command()` |
| `commands/train.py` | Training command |
| `commands/backtest.py` | Backtesting command |
| `commands/walkforward.py` | Walk-forward optimization |
| `commands/autotrade.py` | Live/paper trading |
| `commands/adapt.py` | Model adaptation |
| `commands/evaluate.py` | Model evaluation |

---

## Command Handler Signature

All command handlers MUST follow this signature:

```python
def execute_<command>(
    system: LeapTradingSystem,
    args: argparse.Namespace,
    config: SystemConfig
) -> int:
    """
    Execute the <command> command.

    Args:
        system: The trading system instance (lazy-loads components)
        args: Parsed command-line arguments
        config: System configuration

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Implementation
    return 0
```

---

## Configuration Resolution

Configuration resolves in this order (later overrides earlier):

1. **Defaults**: Hardcoded in dataclass definitions
2. **Config files**: `--config`, `--training-config`, etc.
3. **CLI arguments**: Direct CLI flags

```python
# Example: epochs resolution
# Default: 100 (in TransformerConfig)
# Config file: 50 (if specified)
# CLI: --epochs 25 (overrides all)
final_epochs = args.epochs if args.epochs else config.transformer.epochs
```

### Modular Config Loading

Use modular config loaders:

```python
from config import (
    load_training_config,    # Returns (TransformerConfig, PPOConfig, device, seed)
    load_data_config,        # Returns DataConfig
    load_backtest_config,    # Returns BacktestConfig
    load_risk_config,        # Returns RiskConfig
    load_auto_trader_config, # Returns AutoTraderConfig
)

# In command handler
if args.training_config:
    transformer_cfg, ppo_cfg, device, seed = load_training_config(args.training_config)
```

---

## LeapTradingSystem (Lazy Loading)

`LeapTradingSystem` uses lazy-loading to only initialize needed components:

```python
class LeapTradingSystem:
    @property
    def data_pipeline(self) -> DataPipeline:
        if self._data_pipeline is None:
            self._data_pipeline = DataPipeline(self.config.data)
        return self._data_pipeline

    @property
    def predictor(self) -> TransformerPredictor:
        if self._predictor is None:
            self._predictor = TransformerPredictor(...)
        return self._predictor
```

Access components via properties, NOT direct attributes:

```python
# Correct
data = system.data_pipeline.fetch_historical_data(...)

# Wrong (bypasses lazy loading)
data = system._data_pipeline.fetch_historical_data(...)
```

---

## Data Saving Pattern

Commands that process data should support `--save-data`:

```python
if args.save_data:
    from utils.data_saver import save_pipeline_data, generate_run_id

    run_id = generate_run_id()
    save_pipeline_data(
        run_id=run_id,
        raw_data=market_data,
        features=features,
        symbol=args.symbol,
        timeframe=args.timeframe
    )
    logger.info(f"Data saved to data/{run_id}/")
```

---

## MLflow Context Management

MLflow tracking must be properly managed:

```python
from utils.mlflow_tracker import create_tracker

tracker = create_tracker(experiment_name=f"training_{args.symbol}")

with tracker.start_run(run_name="my_run"):
    # Training code
    tracker.log_params({"epochs": args.epochs})
    tracker.log_metrics({"loss": final_loss})
```

---

## Test Patching

For test compatibility, patch at `main.py` level:

```python
# In tests/test_cli.py

# Correct - patch at main module
@patch('main.TransformerPredictor', MockPredictor)
def test_train_command():
    ...

# Wrong - patch at source module
@patch('models.transformer.TransformerPredictor', MockPredictor)
def test_train_command():
    ...
```

This works because `main.py` re-exports from cli package.

---

## Adding New Commands

1. Create `cli/commands/<command>.py`
2. Implement `execute_<command>(system, args, config)`
3. Add to registry in `cli/commands/__init__.py`
4. Add parser arguments in `cli/parser.py`
5. Update `main.py` if backward compat needed
6. Add tests in `tests/test_cli.py`

```python
# cli/commands/mycommand.py
def execute_mycommand(system: LeapTradingSystem, args, config) -> int:
    logger.info("Running my command")
    # Implementation
    return 0

# cli/commands/__init__.py
COMMAND_REGISTRY = {
    ...
    'mycommand': execute_mycommand,
}
```

---

## Error Handling

Commands should return exit codes:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Configuration error |
| 4 | Runtime error |

```python
def execute_command(system, args, config) -> int:
    try:
        # Implementation
        return 0
    except ConfigurationError:
        logger.error("Invalid configuration")
        return 3
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        return 1
```

---

## DO NOT

- Access `LeapTradingSystem._*` private attributes directly
- Skip config resolution order (defaults → files → CLI)
- Patch at source module in tests (patch at main.py)
- Create commands without adding to registry
- Forget MLflow context cleanup
