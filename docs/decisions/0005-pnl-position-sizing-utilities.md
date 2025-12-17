# ADR 0005: PnL and Position Sizing Utilities

## Status
Accepted

## Context
The Architecture Audit (docs/ARCHITECTURE_AUDIT.md) identified critical code duplication:
- PnL calculations duplicated in 4 files
- Position sizing logic duplicated in 3 files

This led to potential inconsistencies and maintenance burden.

## Decision
Create two centralized utilities:

**`utils/pnl_calculator.py`:**
- `calculate_pnl()` - Realized PnL for closed positions
- `calculate_unrealized_pnl()` - Unrealized PnL for open positions
- `calculate_pnl_percent()` - PnL as percentage

**`utils/position_sizing.py`:**
- `calculate_risk_based_size()` - Risk-based sizing with stop loss
- `calculate_percentage_size()` - Percentage-of-balance sizing
- `apply_position_limits()` - Leverage and max size constraints

**Integration Pattern:**
1. When `RiskManager` is available, use `RiskManager.calculate_position_size()`
2. Fall back to `utils/position_sizing.py` when no RiskManager
3. See `evaluation/backtester.py:_calculate_position_size()` for reference

## Consequences

**Positive:**
- Single source of truth for calculations
- Consistent results across modules
- Easier unit testing
- Documented with examples

**Negative:**
- Modules must import utilities
- All call sites updated (completed in PR #71)

## Code References
- `utils/pnl_calculator.py` - PnL calculations
- `utils/position_sizing.py` - Position sizing
- `evaluation/backtester.py:133-173` - Usage pattern
