# Utils Module Context

The `utils/` directory contains centralized utilities. See ADR-0001 for rationale.

## Utility Overview

| File | Purpose | Key Functions |
|------|---------|---------------|
| `device.py` | Device management | `resolve_device()` |
| `checkpoint.py` | Model persistence | `save_checkpoint()`, `load_checkpoint()` |
| `pnl_calculator.py` | PnL calculations | `calculate_pnl()`, `calculate_unrealized_pnl()` |
| `position_sizing.py` | Position sizing | `calculate_risk_based_size()`, `apply_position_limits()` |
| `logging_config.py` | Logging setup | `setup_logging()` |
| `mlflow_tracker.py` | Experiment tracking | `MLflowTracker`, `create_tracker()` |

## When to Use Each Utility

### PnL Calculations
```python
from utils.pnl_calculator import calculate_pnl, calculate_unrealized_pnl

# Closed position
pnl = calculate_pnl(entry_price, exit_price, size, direction='long')

# Open position
unrealized = calculate_unrealized_pnl(entry_price, current_price, size, 'long')
```

### Position Sizing
```python
from utils.position_sizing import (
    calculate_risk_based_size,
    calculate_percentage_size,
    apply_position_limits
)

# Risk-based sizing
size = calculate_risk_based_size(balance, risk_per_trade, stop_loss_pips, pip_value)

# Apply limits
size = apply_position_limits(size, balance, leverage, entry_price, max_position_size)
```

### Checkpoints
```python
from utils.checkpoint import save_checkpoint, load_checkpoint

checkpoint = load_checkpoint(path, device)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Adding New Utilities

1. Follow existing patterns (see `pnl_calculator.py` for simple example)
2. Add docstrings with Examples section
3. Add `__all__` export list
4. Update this CLAUDE.md
5. Update root CLAUDE.md Centralized Utilities table
6. Create ADR if significant pattern (see `docs/decisions/`)
