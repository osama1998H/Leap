# Architecture Mismatch Report

This report analyzes the Leap trading system architecture to identify where feature implementations deviate from the dominant patterns established in the codebase.

## Executive Summary

The codebase demonstrates strong architectural consistency in several areas (logging, configuration, base class inheritance) but has notable deviations in:
1. **Position sizing API inconsistency** (Critical)
2. **TradingError hierarchy underutilization** (Medium)
3. **Metrics calculation duplication** (Low)
4. **Checkpoint utility adoption gaps** (Low)

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

### CRITICAL: Position Sizing API Inconsistency

**Location:** `core/order_manager.py:429-435`

**Issue:** `OrderManager._calculate_position_params()` calls `RiskManager.calculate_position_size()` with an incompatible signature.

**Actual call:**
```python
volume = self.risk_manager.calculate_position_size(
    account_balance=account.balance,
    risk_percent=risk_percent,
    stop_loss_pips=sl_pips,
    pip_value=self._get_pip_value(signal.symbol, symbol_info)
)
```

**Expected signature (from `core/risk_manager.py:189-193`):**
```python
def calculate_position_size(
    self,
    entry_price: float,
    stop_loss_price: float,
    volatility: Optional[float] = None
) -> float:
```

**Compliant usage (from `evaluation/backtester.py:155-158`):**
```python
size = self.risk_manager.calculate_position_size(
    entry_price=entry_price,
    stop_loss_price=stop_loss_price
)
```

**Impact:** This is a **runtime bug** - the OrderManager will fail at runtime when trying to use RiskManager because the method signature doesn't match.

**Recommendation:** Refactor `OrderManager._calculate_position_params()` to:
1. Calculate `entry_price` and `stop_loss_price` (price levels, not pips)
2. Call `self.risk_manager.calculate_position_size(entry_price, stop_loss_price)`

---

### MEDIUM: TradingError Hierarchy Underutilization

**Defined exceptions (in `core/trading_types.py:28-101`):**
- `TradingError` (base)
- `InsufficientFundsError`
- `OrderRejectedError`
- `PositionError`
- `BrokerConnectionError`
- `DataPipelineError`
- `RiskLimitExceededError`

**Current usage:** Only defined, not actively used throughout the codebase.

**Locations with generic exception handling:**
- `core/order_manager.py:535` - Uses generic `except Exception`
- `evaluation/backtester.py:734` - Uses `except Exception` with `logger.exception()`

**Recommendation:**
1. Replace generic exceptions with domain-specific `TradingError` subclasses
2. In `OrderManager`, raise `OrderRejectedError` for validation failures
3. In `MT5BrokerGateway`, raise `BrokerConnectionError` for connection issues
4. Add catch blocks for `TradingError` hierarchy in calling code

---

### MEDIUM: Inconsistent Risk Validation Flow

**Issue:** Different modules implement risk validation differently.

**Pattern A - Full validation (Correct):**
`core/order_manager.py:205-225`:
```python
should_trade, reason = self.risk_manager.should_take_trade(
    entry_price=entry_price,
    stop_loss_price=sl,
    take_profit_price=tp,
    direction=direction
)
```

**Pattern B - Partial/No validation:**
`core/trading_env.py:205-232` - Opens positions without `should_take_trade()` validation

`core/live_trading_env.py:368-391` - Opens positions without full risk validation

**Recommendation:**
1. Add `should_take_trade()` call before opening positions in `TradingEnvironment._open_position()`
2. Add `should_take_trade()` call before opening positions in `LiveTradingEnvironment._open_position()`

---

### LOW: Metrics Calculation Duplication

**Issue:** Both `BaseTradingEnvironment` and `Backtester` calculate similar metrics independently.

**Location 1:** `core/trading_env_base.py:290-313`
- Delegates to `MetricsCalculator` for Sharpe/Sortino

**Location 2:** `evaluation/backtester.py:489-598`
- Calculates metrics inline in `_calculate_results()`

**Recommendation:**
Refactor `Backtester._calculate_results()` to use `MetricsCalculator` for consistency:
```python
metrics_calc = MetricsCalculator()
sharpe = metrics_calc.sharpe_ratio(returns)
sortino = metrics_calc.sortino_ratio(returns)
```

---

### LOW: Checkpoint Save Method Divergence

**Issue:** `ModelTrainer` uses custom save logic instead of the standardized checkpoint utilities.

**Location:** `training/trainer.py:364-392`

**Current implementation:**
```python
def _save_predictor_checkpoint(self, results: Dict):
    # Saves via predictor.save(path) - delegates to model
    self.predictor.save(path)
    # Saves info as separate JSON
    with open(info_path, 'w') as f:
        json.dump({...}, f)
```

**Standard pattern (from CLAUDE.md):**
```python
from utils.checkpoint import save_checkpoint, TrainingHistory, CheckpointMetadata

training_history = TrainingHistory(train_losses=losses, val_losses=val_losses)
metadata = CheckpointMetadata(model_type='transformer', input_dim=128)
save_checkpoint(path, model.state_dict(), optimizer.state_dict(), config, training_history, metadata)
```

**Recommendation:**
While models themselves correctly use the checkpoint utilities, the `ModelTrainer` could use a unified approach to save metadata alongside model checkpoints.

---

### LOW: Paper Trading Position Sizing Divergence

**Issue:** `LiveTradingEnvironment._execute_paper_trade()` calculates position size inline instead of using RiskManager.

**Location:** `core/live_trading_env.py:469-485`

```python
# Current inline calculation
risk_amount = self._paper_balance * self.risk_per_trade
pip_value = symbol_info.trade_tick_value * (pip_size / symbol_info.trade_tick_size)
volume = risk_amount / (self.default_sl_pips * pip_value)
```

**Standard pattern:**
```python
if self.risk_manager is not None:
    volume = self.risk_manager.calculate_position_size(entry_price, stop_loss)
else:
    # Fallback inline calculation
```

**Recommendation:**
Use RiskManager when available, fall back to inline calculation only when RiskManager is not configured (matches `Backtester` pattern at line 153-162).

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

| Boundary | Issue | Location |
|----------|-------|----------|
| `RiskManager` | Inconsistent method signatures | `order_manager.py:429` |
| `MetricsCalculator` | Duplicated inline in Backtester | `backtester.py:489-598` |

---

## Communication/Data Flow Analysis

### Correct Patterns

1. **Environment -> RiskManager**: Calls `on_position_opened()` / `on_position_closed()` with notional value
   - `core/trading_env.py:238-240, 271-274`
   - `core/live_trading_env.py` (via OrderManager)
   - `evaluation/backtester.py:381-384, 425-427`

2. **Trainer -> MLflow**: Uses callback pattern for metrics logging
   - `training/trainer.py:86-93, 158-177`

### Inconsistent Patterns

1. **OrderManager -> RiskManager**: Uses wrong method signature
2. **Paper trading**: Bypasses RiskManager for position sizing

---

## Cross-Cutting Concerns Analysis

### Logging (Consistent)
All modules use `logger = logging.getLogger(__name__)` pattern.

### Error Handling (Inconsistent)
- Domain exceptions defined but not used
- Generic `except Exception` prevalent
- No structured error propagation

### Configuration (Consistent)
- `EnvConfig` used for environment configuration
- Models use config dicts with sensible defaults

---

## Priority Recommendations

### High Priority (Bug Fixes)
1. **Fix `OrderManager.calculate_position_size` call** - This is a runtime bug
   - File: `core/order_manager.py:429-435`
   - Change to: `calculate_position_size(entry_price, stop_loss_price)`

### Medium Priority (Consistency)
2. **Add risk validation to trading environments**
   - Files: `core/trading_env.py`, `core/live_trading_env.py`
   - Add `should_take_trade()` before `_open_position()`

3. **Use TradingError hierarchy**
   - Replace generic exceptions with domain-specific errors

### Low Priority (Code Quality)
4. **Unify metrics calculation**
   - Refactor `Backtester` to use `MetricsCalculator`

5. **Paper trading risk integration**
   - Use RiskManager in `LiveTradingEnvironment._execute_paper_trade()`

---

## Summary Table

| Feature | Follows Pattern | Abstraction | Data Flow | Cross-Cutting | Priority |
|---------|-----------------|-------------|-----------|---------------|----------|
| TradingEnvironment | Yes | Yes | Yes | Yes | N/A |
| LiveTradingEnvironment | Yes | Yes | Partial | Yes | Low |
| Backtester | Yes | Partial | Yes | Yes | Low |
| OrderManager | No (API mismatch) | Yes | Partial | Yes | **High** |
| PPOAgent | Yes | Yes | Yes | Yes | N/A |
| Transformer | Yes | Yes | Yes | Yes | N/A |
| Trainer | Yes | Yes | Yes | Yes | N/A |
| RiskManager | Yes | Yes | Yes | Yes | N/A |
| DataPipeline | Yes | Yes | Yes | Yes | N/A |

---

*Report generated: 2025-12-10*
*Codebase analyzed: Leap Trading System*
