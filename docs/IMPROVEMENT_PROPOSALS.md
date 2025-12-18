# Leap Trading System - Improvement Proposals

**Date:** 2025-12-17
**Analysis Based On:** CLAUDE.md files, ADRs, Architecture docs, and codebase exploration

---

## Executive Summary

The Leap Trading System is a well-architected AI-powered forex trading platform with strong foundational patterns. The codebase demonstrates excellent consistency in logging, device management, configuration, and checkpoint handling. However, several areas present opportunities for improvement to enhance robustness, maintainability, and production readiness.

### Key Improvement Areas

| Priority | Category | Items | Estimated Effort |
|----------|----------|-------|------------------|
| **Critical** | Test Coverage | 5 | 2-3 days |
| **High** | Architecture | 4 | 2-3 days |
| **Medium** | Code Quality | 6 | 1-2 days |
| **Low** | Performance | 4 | 1-2 days |

---

## 1. Critical: Test Coverage Gaps

### 1.1 Missing Unit Tests for Core Models

**Current State:** The `models/` directory contains critical AI components (`TransformerPredictor`, `PPOAgent`) with **no dedicated unit tests**.

**Impact:**
- Model changes cannot be validated without full integration tests
- Regressions in prediction/action logic go undetected
- Harder to refactor model internals

**Proposed Solution:**
```
tests/
├── test_transformer.py      # NEW: Test TransformerPredictor
│   ├── test_forward_pass_shape()
│   ├── test_quantile_outputs()
│   ├── test_online_learning_update()
│   └── test_checkpoint_save_load()
│
└── test_ppo_agent.py        # NEW: Test PPOAgent
    ├── test_action_selection()
    ├── test_experience_buffer()
    ├── test_gae_computation()
    └── test_policy_update()
```

**Files to Create:**
- `tests/test_transformer.py`
- `tests/test_ppo_agent.py`

---

### 1.2 Missing Tests for Trading Environments (Partially Complete)

**Current State:** `TradingEnvironment` is only tested through integration tests. `LiveTradingEnvironment` now has dedicated tests.

**Impact:**
- Reward calculation edge cases not verified
- TradingEnvironment position management logic not unit tested

**Completed:**
- `tests/test_live_trading_env.py` ✅ (11 tests) - Tests LiveTradingEnvironment with:
  - PaperBrokerGateway integration
  - Mocked BrokerGateway protocol compliance
  - Position synchronization
  - Broker abstraction validation

**Still Needed:**
```
tests/
└── test_trading_env.py           # NEW: Test TradingEnvironment
    ├── test_step_buy_action()
    ├── test_step_sell_action()
    ├── test_reward_calculation()
    ├── test_position_close()
    └── test_observation_space()
```

---

### 1.3 Missing Tests for AutoTrader

**Current State:** `AutoTrader` is a critical production component with **no test coverage**.

**Impact:**
- Live trading behavior cannot be verified without real MT5
- Signal combination logic untested
- State machine transitions not validated

**Proposed Solution:**
```python
# tests/test_auto_trader.py (NEW)
class MockMT5BrokerGateway:
    """Mock broker for testing AutoTrader without MT5."""
    ...

class TestAutoTrader:
    def test_signal_combination_buy_agreement(self)
    def test_signal_combination_buy_contradiction(self)
    def test_state_machine_transitions(self)
    def test_daily_loss_limit_enforcement(self)
    def test_trading_hours_filter(self)
```

---

### 1.4 Missing Tests for Utilities

**Current State:** New utilities added in the recent audit (`pnl_calculator.py`, `position_sizing.py`) lack dedicated tests.

**Impact:**
- PnL calculation edge cases (zero size, extreme prices) not verified
- Position sizing limits not tested at boundaries

**Proposed Solution:**
```
tests/
├── test_pnl_calculator.py     # NEW
│   ├── test_long_pnl()
│   ├── test_short_pnl()
│   ├── test_zero_size()
│   └── test_unrealized_pnl()
│
└── test_position_sizing.py    # NEW
    ├── test_risk_based_sizing()
    ├── test_percentage_sizing()
    ├── test_leverage_limits()
    └── test_max_position_limits()
```

---

### 1.5 Create Test Fixtures Module

**Current State:** Test files duplicate OHLCV generation and mock data creation.

**Proposed Solution:**
```python
# tests/conftest.py (NEW)
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_ohlcv():
    """Generate realistic OHLCV data for testing."""
    ...

@pytest.fixture
def mock_feature_data():
    """Pre-computed features for fast tests."""
    ...

@pytest.fixture
def mock_broker_gateway():
    """Mock MT5 broker for testing."""
    ...
```

---

## 2. High Priority: Architecture Improvements

### 2.1 Broker Interface Abstraction ✅ COMPLETED

**Status:** Implemented in PR #86 and #87

**Implementation:**
- `core/broker_interface.py` - BrokerGateway Protocol, PaperBrokerConfig, create_broker() factory
- `core/paper_broker.py` - PaperBrokerGateway with full position/order management
- `core/mt5_broker.py` - MT5BrokerGateway (Windows only)
- Type hints updated in: PositionSynchronizer, OrderManager, AutoTrader, LiveTradingEnvironment
- Paper trading logic removed from LiveTradingEnvironment (now delegates to broker)
- CLI autotrade command uses create_broker() factory
- Integration tests: `tests/test_live_trading_env.py`, `tests/test_broker_interface.py`

**Benefits Achieved:**
- Cross-platform paper trading
- True dependency injection for brokers
- Easy to add new broker integrations

---

### 2.2 Strategy Pattern for Backtesting ✅ COMPLETED

**Status:** Implemented in PR #88

**Implementation:**
- `core/strategy.py` - `TradingStrategy` ABC, `CombinedPredictorAgentStrategy`, `CallableStrategyAdapter`, `StrategySignal`, `create_strategy()` factory
- `evaluation/backtester.py` - Updated to accept `Union[TradingStrategy, Callable]`, shows deprecation warning for callables
- `core/auto_trader.py` - Accepts optional `TradingStrategy`, auto-creates when predictor/agent available
- `cli/system.py` - Uses `CombinedPredictorAgentStrategy` instead of inline helper functions
- Integration tests: `tests/test_strategy.py::TestBacktesterIntegration`

**Benefits Achieved:**
- Single source of truth for signal combination (eliminated ~140 lines of duplicate code)
- Strategies can be tested independently via ABC interface
- Lifecycle callbacks: `reset()`, `on_trade_opened()`, `on_trade_closed()`
- Backward compatible via `CallableStrategyAdapter`
- See ADR-0011 for full design rationale

---

### 2.3 Feature Engineering Registry ✅ COMPLETED

**Status:** Implemented in ADR-0012

**Implementation:**
- `core/feature_registry.py` - FeatureRegistry singleton with decorator-based registration
- `tests/test_feature_registry.py` - 43 comprehensive unit tests
- `core/data_pipeline.py` - FeatureEngineer now uses registry as facade
- `core/__init__.py` - Exports FeatureRegistry, FeatureCategory, FeatureSpec

**Benefits Achieved:**
- Selective feature computation by category or name
- Easy to add custom features via decorator
- Feature enable/disable without code changes
- Dependency resolution with topological sort
- Better for feature ablation studies

**Usage:**
```python
from core.feature_registry import FeatureRegistry, FeatureCategory

# Get all momentum features
registry = FeatureRegistry.get_instance()
momentum_features = registry.get_feature_names(categories=[FeatureCategory.MOMENTUM])

# Compute specific features
result = registry.compute_all(df, feature_names=['rsi_14', 'macd', 'sma_20'])

# Add custom feature
@FeatureRegistry.register(
    name='my_indicator',
    category=FeatureCategory.CUSTOM,
    dependencies=['close', 'sma_20']
)
def compute_my_indicator(df):
    return df['close'] / df['sma_20'] - 1
```

---

### 2.4 Complete AdaptiveTrainer Integration

**Current State:** `AdaptiveTrainer` is marked as "experimental" and not integrated into CLI.

**Impact:**
- Online learning capabilities not accessible to users
- Experimental code adds maintenance burden

**Proposed Solution:**

Option A: **Full Integration**
```bash
# Add new CLI command
python main.py adaptive-train --symbol EURUSD --online-learning
```

Option B: **Move to Experimental**
```
experimental/
├── adaptive_trainer.py
└── README.md  # Clear documentation of status
```

**Recommendation:** Option A - Complete the integration since online learning is a key differentiator of this system.

---

## 3. Medium Priority: Code Quality

### 3.1 Add Missing Type Annotations

**Locations Needing Types:**
- `core/data_pipeline.py:974-1047` - `prepare_sequences()` internal variables
- `training/online_learning.py:142-236` - `step()` method
- `models/ppo_agent.py` - Multiple methods missing return types
- `evaluation/backtester.py` - Strategy callback signatures

**Proposed Solution:** Add comprehensive type hints following existing patterns.

```python
# Before
def prepare_sequences(self, data, sequence_length, prediction_horizon):
    features = self._compute_features(data)
    ...

# After
def prepare_sequences(
    self,
    data: pd.DataFrame,
    sequence_length: int,
    prediction_horizon: int
) -> Tuple[np.ndarray, np.ndarray, RobustScaler]:
    features: pd.DataFrame = self._compute_features(data)
    ...
```

---

### 3.2 Extract Hardcoded Values to Configuration

**Locations with Magic Numbers:**

| File | Line | Value | Proposed Config |
|------|------|-------|-----------------|
| `data_pipeline.py` | 722 | `200` (buffer) | `DataConfig.fetch_buffer` |
| `data_pipeline.py` | 724 | `500` (min bars) | `DataConfig.min_additional_bars` |
| `online_learning.py` | 134 | `50000` (buffer size) | `AdaptationConfig.experience_buffer_size` |
| `trading_env_base.py` | 295 | `0.5` (clip range) | `EnvConfig.reward_clip_range` |

---

### 3.3 Improve Error Handling in Online Learning

**Current Issue:** Generic exception handling catches all errors including KeyboardInterrupt.

```python
# Current (problematic)
except Exception as e:
    logger.exception(f"Online training error: {e}")
    time.sleep(5)

# Proposed
except (RuntimeError, ValueError, torch.cuda.CudaError) as e:
    logger.exception(f"Online training error: {e}")
    time.sleep(5)
except KeyboardInterrupt:
    logger.info("Online training interrupted by user")
    raise
```

---

### 3.4 Add MT5 Error Details

**Current Issue:** MT5 failures log generic messages without specific error codes.

```python
# Current
if rates is None:
    logger.error(f"Failed to fetch data for {symbol}")
    return None

# Proposed
if rates is None:
    error_code, error_message = mt5.last_error()
    logger.error(f"Failed to fetch data for {symbol}: [{error_code}] {error_message}")
    return None
```

---

### 3.5 Add Memory Bounds to History Tracking

**Current Issue:** `TradingEnvironment.history` grows unboundedly in long episodes.

```python
# Proposed change in trading_env_base.py
from collections import deque

class BaseTradingEnvironment:
    def __init__(self, ..., max_history_length: int = 100000):
        self.history = {
            'balance': deque(maxlen=max_history_length),
            'equity': deque(maxlen=max_history_length),
            'actions': deque(maxlen=max_history_length),
            'rewards': deque(maxlen=max_history_length),
            ...
        }
```

---

### 3.6 Consolidate RSI Implementations

**Current Issue:** Two RSI functions exist: `_compute_rsi()` and `_compute_rsi_wilder()`.

**Proposed Solution:** Keep only Wilder's RSI (the industry standard) and document the method.

---

## 4. Low Priority: Performance Optimizations

### 4.1 Optimize Wilder Smoothing

**Current Issue:** Python loop in `_wilder_smoothing()` called for multiple indicators.

**Proposed Solution:** Use Numba JIT compilation.

```python
from numba import jit

@jit(nopython=True)
def _wilder_smoothing_numba(values: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized Wilder smoothing."""
    n = len(values)
    result = np.empty(n)
    result[:] = np.nan
    ...
    return result
```

**Expected Impact:** 10-50x speedup for feature computation.

---

### 4.2 Add Parquet Support for Data Persistence

**Current Issue:** Data saved as CSV, which is slow for large datasets.

**Proposed Solution:**
```python
# utils/data_saver.py
def save_pipeline_data(..., format: str = 'csv'):
    if format == 'parquet':
        df.to_parquet(path, engine='pyarrow')
    else:
        df.to_csv(path)
```

**Benefits:**
- 5-10x faster I/O
- 3-5x smaller file sizes
- Better type preservation

---

### 4.3 Selective Feature Computation for Multi-Timeframe

**Current Issue:** Computes ALL features for additional timeframes, then selects subset.

```python
# Current (wasteful)
tf_df = tf_engineer.compute_all_features(tf_df)  # 100+ features
selected = tf_df[['ema_20', 'rsi_14', 'atr']]    # Only need 3

# Proposed
tf_df = tf_engineer.compute_features(tf_df, features=['ema_20', 'rsi_14', 'atr'])
```

---

### 4.4 Async Checkpoint Saving

**Current Issue:** Model checkpointing blocks training.

**Proposed Solution:**
```python
# utils/checkpoint.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

_checkpoint_executor = ThreadPoolExecutor(max_workers=1)

def save_checkpoint_async(path, model_state, optimizer_state, ...):
    """Non-blocking checkpoint save."""
    future = _checkpoint_executor.submit(
        save_checkpoint, path, model_state, optimizer_state, ...
    )
    return future
```

---

## 5. Documentation Improvements

### 5.1 Create API Reference

**Proposed:** Auto-generate API documentation using Sphinx or pdoc.

```
docs/
├── api/
│   ├── core.md
│   ├── models.md
│   ├── training.md
│   └── evaluation.md
```

### 5.2 Add Troubleshooting Guide

**Proposed:** Common issues and solutions.

```markdown
# docs/TROUBLESHOOTING.md

## Common Issues

### MT5 Connection Failures
- Symptom: "Failed to connect to MT5"
- Solution: ...

### CUDA Out of Memory
- Symptom: "RuntimeError: CUDA out of memory"
- Solution: ...
```

### 5.3 Add Performance Tuning Guide

**Proposed:** Guide for optimizing training and inference.

```markdown
# docs/PERFORMANCE_TUNING.md

## Training Optimization
- Batch size selection
- GPU memory management
- Feature subset selection

## Inference Optimization
- Model quantization
- Batch inference
```

---

## Implementation Roadmap

### Phase 1: Test Coverage (Week 1)
1. Create `tests/conftest.py` with shared fixtures
2. Add `tests/test_transformer.py`
3. Add `tests/test_ppo_agent.py`
4. Add `tests/test_trading_env.py`
5. Add `tests/test_auto_trader.py` with mocks

### Phase 2: Architecture (Week 2)
1. Create `BrokerGateway` protocol
2. Refactor `MT5BrokerGateway` to implement protocol
3. Create `PaperBrokerGateway` implementation
4. Update `AutoTrader` to use protocol

### Phase 3: Code Quality (Week 3)
1. Add type annotations to flagged locations
2. Extract hardcoded values to configs
3. Improve error handling
4. Add memory bounds

### Phase 4: Polish (Week 4)
1. Performance optimizations
2. Documentation updates
3. Final testing and validation

---

## Conclusion

The Leap Trading System has a solid foundation with excellent patterns in place. The proposed improvements focus on:

1. **Test Coverage** - Reducing risk of regressions
2. **Architecture** - Enabling cross-platform testing and flexibility
3. **Code Quality** - Improving maintainability
4. **Performance** - Optimizing critical paths

Implementing these changes would elevate the health score from 9.2/10 to approximately 9.7/10, with significantly improved production readiness.

---

## References

- [CLAUDE.md](/CLAUDE.md) - Development guidelines
- [ARCHITECTURE.md](/ARCHITECTURE.md) - System architecture
- [ADR Index](/docs/decisions/README.md) - Architectural decisions
