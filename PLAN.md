# Implementation Plan: Architecture Audit Priority 2 & 3 Issues

## Overview

This plan addresses the remaining issues from `docs/ARCHITECTURE_AUDIT.md` after Priority 1 items were resolved.

---

## Priority 2: Medium (Do Soon)

### Issue #5: Add `update_with_trade_result()` to TradingState

**Problem:** Trade statistics update logic is duplicated in:
- `core/trading_env.py:375-380` (`_finalize_position_close()`)
- `core/live_trading_env.py:568-573` (`_close_all_paper_positions()`)
- `core/live_trading_env.py:590-595` (`_close_paper_position()`)
- `core/auto_trader.py:838-843` (uses `session` object)

**Note:** `TradeStatistics` already has `update_from_trade()` at `trading_types.py:347-362`, but `TradingState` doesn't have an equivalent method.

**Solution:**
1. Add `update_with_trade_result(pnl: float)` method to `TradingState` class in `core/trading_types.py:225`
2. Update `_finalize_position_close()` in `trading_env.py` to use the new method
3. Update paper position closing methods in `live_trading_env.py` to use the new method
4. Update `auto_trader.py` session handling to use similar pattern

**Files to modify:**
- `core/trading_types.py` - Add method to TradingState
- `core/trading_env.py` - Use new method in `_finalize_position_close()`
- `core/live_trading_env.py` - Use new method in paper position closing
- `core/auto_trader.py` - Consider using new method or document why session differs

---

### Issue #6: Implement `_calculate_avg_trade_duration()`

**Problem:** Method at `core/trading_env.py:520-523` returns hardcoded `0.0`

```python
def _calculate_avg_trade_duration(self) -> float:
    """Calculate average trade duration."""
    # This would be tracked properly in production
    return 0.0
```

**Solution:**
1. Track trade entry steps and exit steps
2. Maintain a history of trade durations
3. Calculate average from completed trades

**Implementation:**
- Add `_trade_durations: List[int]` to TradingEnvironment state
- Update `_finalize_position_close()` to calculate duration: `current_step - position.entry_time`
- Append duration to `_trade_durations` list
- Compute average in `_calculate_avg_trade_duration()`

**Files to modify:**
- `core/trading_env.py` - Track durations, implement calculation

---

### Issue #7: Document Unused Exception Classes

**Problem:** Exception classes defined in `core/trading_types.py` are only used in docstring examples:
- `InsufficientFundsError` (lines 44-54)
- `OrderRejectedError` (lines 57-65)
- `DataPipelineError` (lines 90-98)
- `RiskLimitExceededError` (lines 101-111)

**Note:** These are part of the public API and may be used by external code.

**Solution:**
Add docstring note marking them as "available for use" rather than removing them:

```python
class InsufficientFundsError(TradingError):
    """
    Raised when account balance is insufficient for a trade.

    Note: This exception is part of the public API and available for use
    in custom implementations and extensions.
    ...
    """
```

**Files to modify:**
- `core/trading_types.py` - Update docstrings for unused exceptions

---

### Issue #8: Create `docs/AUTO_TRADER.md`

**Problem:** `README.md` references `docs/AUTO_TRADER.md` but the file exists at root level as `AUTO_TRADER.md`

**Solution:**
Move `AUTO_TRADER.md` to `docs/AUTO_TRADER.md` (or copy and maintain both)

**Files to modify:**
- Move `/home/user/Leap/AUTO_TRADER.md` to `/home/user/Leap/docs/AUTO_TRADER.md`

---

## Priority 3: Low (When Time Permits)

### Issue #9: Remove `_strategy_type` Parameter

**Problem:** `main.py:441` has unused parameter:
```python
def backtest(
    self,
    market_data,
    _strategy_type: str = 'combined',  # Reserved for future multi-strategy support
    ...
):
```

**Decision needed:** Is multi-strategy support planned?
- If NO: Remove the parameter
- If YES: Keep but update comment with timeline/ticket reference

**Files to modify:**
- `main.py` - Remove parameter or document intent

---

### Issue #10: Document or Integrate AdaptiveTrainer

**Problem:** `AdaptiveTrainer` class (`training/online_learning.py:410-568`) is:
- Imported in `main.py:33`
- Exported in `training/__init__.py:14`
- Documented in `ARCHITECTURE.md:268-275`
- But never actually instantiated or used

**Solution options:**
1. **Document as experimental:** Add note that it's available but not integrated into CLI
2. **Integrate into CLI:** Add `--adaptive` flag to training command
3. **Remove:** If not planned for use, remove the class and update imports

**Recommendation:** Option 1 (Document as experimental) - the class is complete and tested, just needs explicit documentation that it's for programmatic use, not CLI.

**Files to modify:**
- `ARCHITECTURE.md` - Add note about experimental status
- Or `training/online_learning.py` - Add class-level docstring note

---

### Issue #11: Extract Position Closing Logic to Base Class

**Problem:** Position closing logic duplicated across:
- `core/trading_env.py` - `_finalize_position_close()`, `_close_position()`, `_close_position_at_sl_tp()`
- `core/live_trading_env.py` - `_close_paper_position()`, `_close_all_paper_positions()`
- `core/trading_env_base.py` - Has abstract interface but no shared implementation

**Solution:**
After implementing Issue #5 (`TradingState.update_with_trade_result()`), refactor:
1. Add shared `_update_stats_on_close(pnl)` to `TradingEnvironmentBase`
2. Have child classes call parent method after PnL calculation
3. Keep PnL calculation specific to each class (different spread/slippage handling)

**Note:** This depends on Issue #5 being completed first.

**Files to modify:**
- `core/trading_env_base.py` - Add shared statistics update method
- `core/trading_env.py` - Use shared method
- `core/live_trading_env.py` - Use shared method

---

## Implementation Order

Recommended sequence based on dependencies:

1. **Issue #5** - TradingState.update_with_trade_result() (enables #11)
2. **Issue #6** - _calculate_avg_trade_duration() (independent)
3. **Issue #7** - Document exception classes (independent, low effort)
4. **Issue #8** - Move AUTO_TRADER.md (independent, low effort)
5. **Issue #9** - Remove _strategy_type (independent, low effort)
6. **Issue #10** - Document AdaptiveTrainer (independent, low effort)
7. **Issue #11** - Extract position closing logic (depends on #5)

---

## Estimated Effort

| Issue | Effort | Lines Changed |
|-------|--------|---------------|
| #5 | Low | ~30 lines |
| #6 | Low | ~20 lines |
| #7 | Low | ~15 lines |
| #8 | Low | File move |
| #9 | Low | ~2 lines |
| #10 | Low | ~10 lines |
| #11 | Medium | ~50 lines |

**Total estimated time:** 2-4 hours
