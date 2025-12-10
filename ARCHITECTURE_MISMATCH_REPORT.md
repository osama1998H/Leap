# Architecture Mismatch Report

This report analyzes the Leap trading system architecture to identify where feature implementations deviate from the dominant patterns established in the codebase.

## Executive Summary

The codebase demonstrates strong architectural consistency in several areas (logging, configuration, base class inheritance). Previous deviations have been **RESOLVED** as of 2025-12-10:

1. ~~**Position sizing API inconsistency** (Critical)~~ → **RESOLVED**
2. ~~**TradingError hierarchy underutilization** (Medium)~~ → **RESOLVED**
3. ~~**Risk validation missing in trading environments** (Medium)~~ → **RESOLVED**
4. ~~**Metrics calculation duplication** (Low)~~ → **RESOLVED**
5. **Checkpoint utility adoption gaps** (Low) - No action required (models correctly use checkpoint utilities)

---

## Dominant Architectural Patterns

### 1. Logging Pattern
**Standard:** `logger = logging.getLogger(__name__)`

**Compliance:** 100% - All 21 modules consistently use this pattern.

### 2. Device Resolution Pattern
**Standard:** `from utils.device import resolve_device` then `self.device = resolve_device(device)`

**Compliant files:**
- `models/ppo_agent.py`
- `models/transformer.py`
- `training/trainer.py`

### 3. Configuration Pattern
**Standard:** Use `EnvConfig` dataclass from `core/trading_types.py`, with `EnvConfig.from_params()` factory

**Compliant files:**
- `core/trading_env_base.py:70` - Uses `EnvConfig.from_params()` factory
- `core/trading_env.py` - Passes config to base class
- `core/live_trading_env.py` - Passes config to base class

### 4. Trading Environment Pattern
**Standard:** Inherit from `BaseTradingEnvironment`, implement abstract methods

**Compliant files:**
- `core/trading_env.py:23` - `TradingEnvironment(BaseTradingEnvironment)`
- `core/live_trading_env.py:23` - `LiveTradingEnvironment(BaseTradingEnvironment)`

### 5. Checkpoint Pattern
**Standard:** Use `utils/checkpoint.py` with `save_checkpoint()`, `load_checkpoint()`, `TrainingHistory`, `CheckpointMetadata`

**Compliant files:**
- `models/ppo_agent.py:16-21`
- `models/transformer.py:17-22`

### 6. Risk Manager Integration Pattern
**Standard:**
- Use `RiskManager.calculate_position_size(entry_price, stop_loss_price)` for sizing
- Use `RiskManager.should_take_trade(entry_price, stop_loss_price, take_profit_price, direction)` for validation
- Call `risk_manager.on_position_opened(notional)` and `on_position_closed(notional)`

---

## Mismatch Analysis

### ✅ RESOLVED: Position Sizing API Inconsistency

**Location:** `core/order_manager.py:429-435`

**Previous Issue:** `OrderManager._calculate_position_params()` called `RiskManager.calculate_position_size()` with an incompatible signature.

**Resolution (2025-12-10):** Fixed to use correct signature:
```python
volume = self.risk_manager.calculate_position_size(
    entry_price=entry_price,
    stop_loss_price=sl_price
)
```

---

### ✅ RESOLVED: TradingError Hierarchy Underutilization

**Defined exceptions (in `core/trading_types.py:28-101`):**
- `TradingError` (base)
- `InsufficientFundsError`
- `OrderRejectedError`
- `PositionError`
- `BrokerConnectionError`
- `DataPipelineError`
- `RiskLimitExceededError`

**Resolution (2025-12-10):**
- `core/order_manager.py` now imports and catches `TradingError` subclasses
- `evaluation/backtester.py` now catches `TradingError` specifically before generic `Exception`

---

### ✅ RESOLVED: Inconsistent Risk Validation Flow

**Previous Issue:** Different modules implemented risk validation differently.

**Resolution (2025-12-10):**
- `core/trading_env.py:_open_position()` now calls `should_take_trade()` before opening positions
- `core/live_trading_env.py:_open_position()` now calls `should_take_trade()` before opening positions
- Both also use `calculate_position_size()` when RiskManager is available

---

### ✅ RESOLVED: Metrics Calculation Duplication

**Previous Issue:** `Backtester` calculated metrics inline instead of using `MetricsCalculator`.

**Resolution (2025-12-10):**
`evaluation/backtester.py:_calculate_results()` now uses `MetricsCalculator`:
```python
metrics_calc = MetricsCalculator()
total_return = metrics_calc.total_return(equity)
sharpe = metrics_calc.sharpe_ratio(returns)
sortino = metrics_calc.sortino_ratio(returns)
# ... etc
```

---

### LOW: Checkpoint Save Method Divergence (No Action Required)

**Issue:** `ModelTrainer` uses custom save logic instead of the standardized checkpoint utilities.

**Location:** `training/trainer.py:364-392`

**Status:** This is acceptable behavior. The `ModelTrainer` delegates to the model's own save method (`predictor.save()`), which uses the checkpoint utilities. The additional JSON metadata is trainer-specific and doesn't need to follow the checkpoint pattern.

---

### ✅ RESOLVED: Paper Trading Position Sizing Divergence

**Previous Issue:** `LiveTradingEnvironment._execute_paper_trade()` calculated position size inline instead of using RiskManager.

**Resolution (2025-12-10):**
`core/live_trading_env.py:_execute_paper_trade()` now uses RiskManager when available:
```python
if self.risk_manager is not None:
    volume = self.risk_manager.calculate_position_size(
        entry_price=entry_price,
        stop_loss_price=sl
    )
else:
    # Fallback inline calculation
```

---

## Abstraction Boundary Analysis

### Well-Maintained Boundaries

| Boundary | Description | Status |
|----------|-------------|--------|
| `BaseTradingEnvironment` | Abstract base for all trading envs | Properly enforced |
| `EnvConfig` | Environment configuration | Consistently used |
| `MetricsCalculator` | Metrics computation | Used in base env |
| `Trade` / `Position` | Data transfer objects | Properly separated |

### Boundaries with Leakage

| Boundary | Issue | Location | Status |
|----------|-------|----------|--------|
| `RiskManager` | ~~Inconsistent method signatures~~ | `order_manager.py:429` | ✅ **RESOLVED** |
| `MetricsCalculator` | ~~Duplicated inline in Backtester~~ | `backtester.py:489-598` | ✅ **RESOLVED** |

---

## Communication/Data Flow Analysis

### Correct Patterns

1. **Environment -> RiskManager**: Calls `on_position_opened()` / `on_position_closed()` with notional value
   - `core/trading_env.py:238-240, 271-274`
   - `core/live_trading_env.py` (via OrderManager)
   - `evaluation/backtester.py:381-384, 425-427`

2. **Trainer -> MLflow**: Uses callback pattern for metrics logging
   - `training/trainer.py:86-93, 158-177`

### ~~Inconsistent Patterns~~ (All Resolved)

1. ~~**OrderManager -> RiskManager**: Uses wrong method signature~~ → ✅ **FIXED**
2. ~~**Paper trading**: Bypasses RiskManager for position sizing~~ → ✅ **FIXED**

---

## Cross-Cutting Concerns Analysis

### Logging (Consistent)
All modules use `logger = logging.getLogger(__name__)` pattern.

### Error Handling (Improved)
- ✅ Domain exceptions now imported and caught in key modules
- ✅ `OrderManager` catches `TradingError` subclasses
- ✅ `Backtester` catches `TradingError` before generic `Exception`

### Configuration (Consistent)
- `EnvConfig` used for environment configuration
- Models use config dicts with sensible defaults

---

## Priority Recommendations

### ✅ All High/Medium Priority Items Resolved (2025-12-10)

1. ~~**Fix `OrderManager.calculate_position_size` call**~~ → **DONE**
2. ~~**Add risk validation to trading environments**~~ → **DONE**
3. ~~**Use TradingError hierarchy**~~ → **DONE**
4. ~~**Unify metrics calculation**~~ → **DONE**
5. ~~**Paper trading risk integration**~~ → **DONE**

### Remaining Low Priority (No Action Required)
- **Checkpoint Save Method Divergence** - Acceptable as-is (ModelTrainer correctly delegates to model's save method)

---

## Summary Table

| Feature | Follows Pattern | Abstraction | Data Flow | Cross-Cutting | Status |
|---------|-----------------|-------------|-----------|---------------|--------|
| TradingEnvironment | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | **Compliant** |
| LiveTradingEnvironment | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | **Compliant** |
| Backtester | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | **Compliant** |
| OrderManager | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | **Compliant** |
| PPOAgent | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | **Compliant** |
| Transformer | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | **Compliant** |
| Trainer | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | **Compliant** |
| RiskManager | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | **Compliant** |
| DataPipeline | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | **Compliant** |

---

*Report generated: 2025-12-10*
*Last updated: 2025-12-10 (All issues resolved)*
*Codebase analyzed: Leap Trading System*
