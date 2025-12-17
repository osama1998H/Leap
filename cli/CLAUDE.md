# CLI Module Context

The `cli/` directory contains the command-line interface implementation for the Leap Trading System.

## Module Overview

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `__init__.py` | Package exports | `main()`, `LeapTradingSystem`, `initialize_logging` |
| `parser.py` | Argument parsing | `create_parser()`, `initialize_logging()`, `resolve_cli_config()` |
| `system.py` | Main orchestrator | `LeapTradingSystem` class |
| `commands/__init__.py` | Command registry | `execute_command()`, `COMMANDS` dict |
| `commands/train.py` | Train command | `execute_train()` |
| `commands/backtest.py` | Backtest command | `execute_backtest()` |
| `commands/walkforward.py` | Walk-forward command | `execute_walkforward()` |
| `commands/evaluate.py` | Evaluate command | `execute_evaluate()` |
| `commands/autotrade.py` | Auto-trade command | `execute_autotrade()` |

## Critical Patterns

### Backward Compatibility
Imports from `main.py` still work for backward compatibility:
```python
from main import LeapTradingSystem, main, initialize_logging
```

The `main.py` file is a thin wrapper that re-exports from this package.

### Adding New Commands
1. Create `cli/commands/newcmd.py` with `execute_newcmd(system, args, config)`
2. Add to `cli/commands/__init__.py` COMMANDS dict
3. Add parser configuration in `cli/parser.py`
4. Update CLI.md documentation

### Command Handler Signature
All command handlers follow the same signature:
```python
def execute_<command>(system: LeapTradingSystem, args: argparse.Namespace, config: SystemConfig) -> None:
    """Execute the <command> command."""
    ...
```

### Configuration Resolution
CLI arguments are resolved in `parser.py:resolve_cli_config()`:
1. Start with default `SystemConfig`
2. Override with modular config files if provided
3. CLI arguments take final precedence

### Data Saving Pattern (--save-data)
All command handlers support the `--save-data` flag using this pattern:
```python
if getattr(args, 'save_data', False) and market_data is not None:
    from utils.data_saver import save_pipeline_data, generate_run_id
    run_id = generate_run_id("command_name", symbol, timeframe)
    data_source = "MT5" if getattr(system.data_pipeline, 'broker_gateway', None) else "synthetic"
    save_pipeline_data(
        run_id=run_id,
        market_data=market_data,
        base_dir=config.get_path('data'),
        command="command_name",
        n_bars=n_bars,
        data_source=data_source
    )
```
See ADR-0009 for design rationale.

## Common Gotchas

1. **Test Patching**: Tests patch at `main` module level (e.g., `@patch('main.TransformerPredictor')`). The `main.py` wrapper imports these classes to enable patching.

2. **Lazy Loading**: `LeapTradingSystem` uses lazy loading for components. Access via properties triggers initialization.

3. **MLflow Context**: Training commands manage MLflow run context. Always use try/finally for proper cleanup.

4. **AutoTrader Availability**: The `autotrade` command checks `AUTO_TRADER_AVAILABLE` flag (MT5 is Windows-only).
