# ADR 0001: Centralized Utilities Pattern

## Status
Accepted

## Context
The Leap codebase has multiple modules (core/, models/, training/, evaluation/) that need common functionality:
- Device management (CPU/GPU/MPS selection)
- Model checkpointing
- Logging configuration
- PnL calculations
- Position sizing

Without centralization, each module implements these differently, leading to:
- Code duplication
- Inconsistent behavior
- Maintenance burden

## Decision
Create centralized utilities in `utils/` directory:

| Utility | File | Purpose |
|---------|------|---------|
| Device | `device.py` | `resolve_device()` for PyTorch device handling |
| Checkpoint | `checkpoint.py` | Standardized model save/load |
| Logging | `logging_config.py` | Unified logging setup |
| PnL | `pnl_calculator.py` | Consistent PnL calculations |
| Position Sizing | `position_sizing.py` | Position size calculations |

**Rule:** All modules MUST use these utilities instead of implementing their own.

## Consequences

**Positive:**
- Single source of truth for common operations
- Consistent behavior across modules
- Easier testing and maintenance
- Clear documentation location

**Negative:**
- Dependency on utils/ from all modules
- Must update centralized code carefully

## Code References
- `utils/device.py` - Device resolution
- `utils/checkpoint.py` - Model persistence
- `utils/pnl_calculator.py` - PnL calculations
- `utils/position_sizing.py` - Position sizing
