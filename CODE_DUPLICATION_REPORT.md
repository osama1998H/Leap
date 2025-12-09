# Code Duplication and Inconsistency Analysis Report

**Generated:** 2025-12-09
**Updated:** 2025-12-09
**Codebase:** Leap AI/ML Trading System

---

## Executive Summary

This report identifies code duplications and inconsistencies across the Leap codebase. The analysis found **8 major areas** of duplication/inconsistency. **2 HIGH priority issues have been resolved**.

| Category | Severity | Files Affected | Status |
|----------|----------|----------------|--------|
| Device Setup Pattern | ~~High~~ | 4 files | **RESOLVED** - Centralized in `utils/device.py` |
| Logging Initialization | Medium | 12+ files | Pending - Standardize approach |
| Trade/Position Dataclasses | ~~High~~ | 5 files | **RESOLVED** - Consolidated in `trading_types.py` |
| Metrics Calculation | Medium | 3 files | Pending - Create shared metrics module |
| Position Sizing Logic | Medium | 3 files | Pending - Move to RiskManager |
| Risk Validation | Medium | 4 files | Pending - Create validation utilities |
| Config Parameter Handling | Low | 3 files | Pending - Use config dataclasses consistently |
| Model Save/Load Patterns | Low | 3 files | Pending - Standardize approach |

---

## Resolved Issues

### 1. Device Setup Pattern (RESOLVED)

**Solution Implemented:**
- Created `utils/device.py` with centralized device utilities:
  - `get_device()`: Returns best available PyTorch device (cached)
  - `resolve_device(device)`: Resolves device specification to torch.device
  - `get_device_string()`: Returns device as string for configuration
- Updated all affected files to use `resolve_device()`:
  - `models/transformer.py`
  - `models/ppo_agent.py`
  - `training/trainer.py`

**Impact:**
- Eliminated code duplication across 3 files
- Added Apple Silicon MPS support
- Consistent device handling with caching for performance
- Single point of maintenance for device detection logic

### 2. Trade/Position Dataclasses (RESOLVED)

**Solution Implemented:**
- Added consolidated `Trade` dataclass to `core/trading_types.py`:
  - Supports both datetime and int entry_time (flexible)
  - Includes `is_closed` and `is_winning` properties
  - Compatible with backtester and auto_trader use cases
- Added `TradeStatistics` dataclass to `core/trading_types.py`:
  - Reusable statistics with `win_rate`, `profit_factor` properties
  - `update_from_trade()` method for easy updates
- Updated `evaluation/backtester.py` to import Trade from trading_types
- Added `get_trade_statistics()` method to `TradingSession` in auto_trader

**Impact:**
- Eliminated duplicate Trade dataclass in backtester.py
- Consolidated trade statistics into reusable dataclass
- Backward-compatible changes (no breaking changes)
- Single source of truth for trade-related types

---

## Remaining Issues

### 1. Logging Initialization (MEDIUM)

### Issue
Different files use inconsistent patterns for logger initialization.

### Pattern A: Standard module logger (MOST COMMON)
**Files:** `core/risk_manager.py`, `core/auto_trader.py`, `evaluation/backtester.py`, `evaluation/metrics.py`, `training/trainer.py`, `training/online_learning.py`, `core/trading_env_base.py`
```python
import logging
logger = logging.getLogger(__name__)
```

### Pattern B: Custom logging config import
**Files:** `utils/logging_config.py` provides `get_logger()` but it's not consistently used

### Pattern C: Direct logging calls without module logger
Some files make direct `logging.info()` calls without a module-level logger.

### Recommendation
Standardize on Pattern A with enhanced logging config:

```python
# In all files:
import logging
logger = logging.getLogger(__name__)

# Configure once in main.py or utils/logging_config.py
```

**Standardize on:** `logging.getLogger(__name__)` pattern in all modules.

---

### 2. Metrics Calculation (MEDIUM)

### Issue
Similar metric calculations are implemented in multiple places.

### Locations

**`evaluation/metrics.py:151-162` - Sharpe ratio**
```python
def sharpe_ratio(self, returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - self.risk_free_rate / self.periods_per_year
    vol = np.std(returns)
    if vol == 0:
        return 0.0
    return np.mean(excess_returns) / vol * np.sqrt(self.periods_per_year)
```

**`core/trading_env_base.py:277-285` - Sharpe ratio**
```python
def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
```

**`evaluation/backtester.py:477-481` - Sharpe ratio (inline)**
```python
risk_free_rate = 0.02
excess_return = annualized_return - risk_free_rate
sharpe = excess_return / volatility if volatility > 0 else 0
```

### Discrepancies
1. Different assumptions for `periods_per_year` (252 vs 252*24)
2. Different handling of edge cases
3. Calculation approach differs slightly

### Recommendation
Use `MetricsCalculator` from `evaluation/metrics.py` everywhere:

```python
# In trading_env_base.py and backtester.py:
from evaluation.metrics import MetricsCalculator

class SomeClass:
    def __init__(self):
        self._metrics = MetricsCalculator(
            risk_free_rate=0.02,
            periods_per_year=252 * 24  # Configure based on timeframe
        )

    def _calculate_sharpe(self, returns):
        return self._metrics.sharpe_ratio(returns)
```

**Standardize on:** `MetricsCalculator` class with configurable parameters.

---

### 3. Position Sizing Logic (MEDIUM)

### Issue
Position sizing logic is duplicated with variations.

### Locations

**`core/risk_manager.py:189-226` - RiskManager**
```python
def calculate_position_size(self, entry_price, stop_loss_price, volatility=None):
    if self.sizing.method == 'fixed':
        size = self.sizing.fixed_size
    elif self.sizing.method == 'percent':
        risk_amount = self.current_balance * self.sizing.percent_risk
        size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
    elif self.sizing.method == 'kelly':
        size = self._kelly_position_size(entry_price, risk_per_unit)
    elif self.sizing.method == 'volatility':
        # ...
```

**`evaluation/backtester.py:317-327` - Backtester inline**
```python
# Calculate position size based on risk
risk_amount = self.balance * self.risk_per_trade
size = risk_amount / (safe_stop_pips * self.pip_value * entry_price)

# Apply leverage limit
max_size = (self.balance * self.leverage) / entry_price
size = min(size, max_size)
```

**`core/order_manager.py:401-416` - OrderManager**
```python
if self.risk_manager:
    volume = self.risk_manager.calculate_position_size(...)
else:
    # Simple position sizing: risk_amount / (sl_pips * pip_value)
    risk_amount = account.balance * risk_percent
    pip_value = self._get_pip_value(signal.symbol, symbol_info)
    if pip_value > 0 and sl_pips > 0:
        volume = risk_amount / (sl_pips * pip_value)
```

### Recommendation
Always delegate to RiskManager:

```python
# In backtester.py:
def __init__(self, ..., risk_manager=None):
    self.risk_manager = risk_manager or RiskManager(
        initial_balance=initial_balance,
        sizing=PositionSizing(method='percent', percent_risk=risk_per_trade)
    )

def _calculate_size(self, entry_price, stop_loss_price):
    return self.risk_manager.calculate_position_size(entry_price, stop_loss_price)
```

**Standardize on:** `RiskManager.calculate_position_size()` for all sizing.

---

### 4. Risk Validation Logic (MEDIUM)

### Issue
Trade validation/risk checking is implemented differently across files.

### Locations

**`core/risk_manager.py:314-373` - should_take_trade()**
```python
def should_take_trade(self, entry_price, stop_loss_price, take_profit_price, direction):
    # Validates direction, price logic, risk/reward, position size, max positions
    if not self.state.is_trading_allowed:
        return False, f"Trading halted: {self.state.halt_reason}"
    # ... detailed validation
```

**`core/order_manager.py:327-364` - _validate_signal()**
```python
def _validate_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
    # Check confidence, broker connection, symbol, spread, account
    if self.risk_manager:
        if not self.risk_manager.should_take_trade():  # Different signature!
            return False, "Risk manager rejected trade"
```

**`evaluation/backtester.py:250-258` - Inline validation**
```python
# Check cooldown constraint
bars_since_last_trade = bar_index - self._last_trade_bar
if bars_since_last_trade < self.min_bars_between_trades:
    if action in ('buy', 'sell'):
        return  # Skip trade due to cooldown
```

### Issues
1. `RiskManager.should_take_trade()` requires 4 parameters but OrderManager calls it with none
2. Different validation checks in different places
3. No unified validation interface

### Recommendation
Create a unified validation system:

```python
# core/trade_validator.py

@dataclass
class TradeValidationRequest:
    symbol: str
    direction: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    confidence: float = 1.0

@dataclass
class ValidationResult:
    is_valid: bool
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

class TradeValidator:
    def __init__(self, risk_manager=None, broker=None):
        self.risk_manager = risk_manager
        self.broker = broker

    def validate(self, request: TradeValidationRequest) -> ValidationResult:
        # Unified validation logic
        pass
```

**Standardize on:** Unified `TradeValidator` class with consistent interface.

---

### 5. Config Parameter Handling (LOW)

### Issue
Some classes accept both config dataclass AND individual parameters, leading to complex initialization.

### Location

**`core/trading_env_base.py:31-88`**
```python
def __init__(
    self,
    config: Optional[EnvConfig] = None,
    initial_balance: float = 10000.0,
    commission: float = 0.0001,
    # ... 8 more individual params
):
    if config is not None:
        self.initial_balance = config.initial_balance
        # ... copy all config fields
    else:
        self.initial_balance = initial_balance
        # ... use individual params
```

### Recommendation
Use factory pattern or require config only:

```python
# Option 1: Factory method
class BaseTradingEnvironment:
    def __init__(self, config: EnvConfig):
        self.config = config
        # Use config directly

    @classmethod
    def from_params(cls, initial_balance=10000.0, commission=0.0001, ...):
        config = EnvConfig(initial_balance=initial_balance, commission=commission, ...)
        return cls(config)

# Option 2: Config only with defaults
class BaseTradingEnvironment:
    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
```

**Standardize on:** Config dataclasses only, with factory methods for convenience.

---

### 6. Model Save/Load Patterns (LOW)

### Issue
Different approaches to model serialization across files.

### Locations

**`models/transformer.py:289-310` - save() method**
```python
def save(self, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        'config': self.config.__dict__,
        'scaler_params': {...}
    }
    torch.save(checkpoint, path)
```

**`models/ppo_agent.py:306-323` - save() method**
```python
def save(self, path: str):
    checkpoint = {
        'network_state_dict': self.network.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'config': {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            # ... individual fields
        }
    }
    torch.save(checkpoint, path)
```

### Differences
1. TransformerPredictor saves `config.__dict__`, PPOAgent saves individual fields
2. Different key names (`model_state_dict` vs `network_state_dict`)
3. TransformerPredictor has `scaler_params`, PPOAgent doesn't

### Recommendation
Create a base checkpoint interface:

```python
# utils/checkpointing.py

@dataclass
class ModelCheckpoint:
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Optional[Dict[str, Any]]
    config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

def save_checkpoint(path: str, checkpoint: ModelCheckpoint):
    torch.save(asdict(checkpoint), path)

def load_checkpoint(path: str) -> ModelCheckpoint:
    data = torch.load(path, map_location='cpu')
    return ModelCheckpoint(**data)
```

**Standardize on:** `ModelCheckpoint` dataclass with consistent structure.

---

## Priority Implementation Order

1. **HIGH Priority - COMPLETED**
   - ~~Device setup centralization~~ ✓
   - ~~Trade/Position dataclass consolidation~~ ✓

2. **MEDIUM Priority (Next)**
   - Metrics calculation unification
   - Position sizing consolidation
   - Risk validation unification

3. **LOW Priority**
   - Config parameter handling
   - Model save/load patterns
   - Logging standardization

---

## Migration Strategy

### Phase 1: Create shared utilities without breaking changes - COMPLETED
1. ~~Create `utils/device.py` with `get_device()`~~ ✓
2. Create `utils/checkpointing.py` with checkpoint utilities
3. ~~Enhance `core/trading_types.py` with consolidated dataclasses~~ ✓

### Phase 2: Gradual migration - IN PROGRESS
1. ~~Update imports in each file to use new utilities~~ ✓
2. Add deprecation warnings to old patterns
3. Update tests to verify behavior consistency

### Phase 3: Cleanup
1. Remove deprecated code paths
2. Update documentation
3. Add linting rules to enforce patterns

---

## Appendix: File-by-File Summary

| File | Duplications Found | Status |
|------|-------------------|--------|
| `models/transformer.py` | ~~Device setup~~, save pattern | Device setup resolved |
| `models/ppo_agent.py` | ~~Device setup~~, save pattern | Device setup resolved |
| `training/trainer.py` | ~~Device setup~~ | **Fully resolved** |
| `core/trading_types.py` | ~~Base for consolidation~~ | **Now authoritative** |
| `core/trading_env_base.py` | Metrics calc, config handling | Pending |
| `core/risk_manager.py` | Position sizing (authoritative) | Pending |
| `core/order_manager.py` | Position sizing fallback, validation | Pending |
| `core/auto_trader.py` | ~~Trade statistics~~ | **Uses TradeStatistics** |
| `evaluation/backtester.py` | ~~Trade class~~, metrics, position sizing | Trade class resolved |
| `evaluation/metrics.py` | Metrics (authoritative) | Pending |
