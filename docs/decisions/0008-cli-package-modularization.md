# ADR 0008: CLI Package Modularization

## Status
Accepted

## Context

The `main.py` file had grown to approximately 1970 lines, combining multiple concerns:

1. **Argument parsing** (~160 lines): CLI argument definitions and configuration
2. **Configuration loading** (~100 lines): Config file loading and CLI override resolution
3. **LeapTradingSystem class** (~1250 lines): Core orchestrator with all trading system components
4. **Command execution** (~350 lines): Implementation of train, backtest, evaluate, walkforward, autotrade commands
5. **Logging initialization** (~40 lines): Logging setup with config support

This monolithic structure created several problems:

- **Difficult navigation**: Finding specific functionality required extensive scrolling
- **Hard to maintain**: Changes to one command risked affecting others
- **Poor separation of concerns**: Business logic mixed with CLI parsing
- **Testing challenges**: Tests had to patch deeply into a single large module
- **AI tool limitations**: Large files are harder for AI tools to process effectively

## Decision

Refactor `main.py` into a modular `cli/` package with clear separation of concerns:

```
cli/
├── __init__.py           # Package exports and main() entry point
├── CLAUDE.md             # Module-specific documentation
├── parser.py             # Argument parsing and config resolution
├── system.py             # LeapTradingSystem class
└── commands/
    ├── __init__.py       # Command registry and dispatch
    ├── train.py          # train command
    ├── backtest.py       # backtest command
    ├── walkforward.py    # walkforward command
    ├── evaluate.py       # evaluate command
    └── autotrade.py      # autotrade command
```

### Key Design Decisions

1. **Backward compatibility wrapper**: Keep `main.py` as a thin wrapper that re-exports from `cli/`, ensuring:
   - `from main import LeapTradingSystem` still works
   - `@patch('main.TransformerPredictor')` test patterns still work
   - Existing scripts and documentation remain valid

2. **Command handler pattern**: Each command is a function with signature:
   ```python
   def execute_<command>(system, args, config, resolved) -> None
   ```

3. **Configuration resolution**: A single `resolve_cli_config()` function handles all config loading and CLI override logic, returning both the config object and resolved values.

4. **Strategy logic stays in system**: Per user preference, complex strategy logic (like backtest signal combination) remains in `LeapTradingSystem.backtest()` rather than being extracted to command handlers.

### File Sizes

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~55 | Backward compatibility wrapper |
| `cli/__init__.py` | ~75 | Package exports and main() |
| `cli/parser.py` | ~275 | Argument parsing and config resolution |
| `cli/system.py` | ~1100 | LeapTradingSystem class |
| `cli/commands/__init__.py` | ~55 | Command registry |
| `cli/commands/train.py` | ~110 | Train command |
| `cli/commands/backtest.py` | ~105 | Backtest command |
| `cli/commands/walkforward.py` | ~55 | Walk-forward command |
| `cli/commands/evaluate.py` | ~65 | Evaluate command |
| `cli/commands/autotrade.py` | ~170 | Auto-trade command |

All files are now under 1200 lines, with most under 200 lines.

## Consequences

### Positive

- **Clear separation of concerns**: Each file has a single responsibility
- **Easier navigation**: Developers can quickly find and modify specific functionality
- **Better testability**: Command handlers can be tested independently
- **Extensibility**: New commands can be added without modifying existing files
- **AI-friendly**: Smaller files are easier for AI tools to process and understand
- **Maintained compatibility**: Existing code and tests continue to work

### Negative

- **More files to navigate**: Developers must understand the package structure
- **Slightly longer import paths**: Internal code uses `from cli.system import LeapTradingSystem`
- **Initial learning curve**: New contributors must understand the modular structure

### Test Impact

**No test changes required** because:
- `main.py` re-exports `LeapTradingSystem`, `main`, `initialize_logging`
- `main.py` imports commonly-patched classes (`TransformerPredictor`, `PPOAgent`, etc.)
- Test patterns like `@patch('main.TransformerPredictor')` continue to work

## Implementation Notes

1. The `cli/CLAUDE.md` file documents module-specific patterns and conventions
2. Command handlers follow a consistent signature for easy dispatch
3. The backward compatibility wrapper in `main.py` is intentionally minimal
4. Strategy logic in `LeapTradingSystem.backtest()` remains inline (user preference)

## Related Decisions

- [ADR-0007: Modular Configuration System](0007-modular-config-system.md) - Config loading patterns
- [ADR-0001: Centralized Utilities](0001-centralized-utilities.md) - Utility organization patterns
