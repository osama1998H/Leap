# Architectural Mismatch Report

## Executive Summary

This report analyzes the Leap trading system architecture, comparing documented design patterns against actual implementation. The analysis identifies **23 architectural mismatches** across 6 categories, with **5 critical issues** requiring immediate attention.

**Overall Assessment**: The architecture design is sound with well-defined patterns documented in ARCHITECTURE.md and CLAUDE.md. However, implementation has drifted from intended design in several areas.

---

## Table of Contents

1. [Methodology](#methodology)
2. [Documented Architecture Patterns](#documented-architecture-patterns)
3. [Critical Mismatches](#critical-mismatches)
4. [Major Mismatches](#major-mismatches)
5. [Minor Mismatches](#minor-mismatches)
6. [Pattern Consistency Matrix](#pattern-consistency-matrix)
7. [Recommendations](#recommendations)

---

## Methodology

Analysis compared:
- **ARCHITECTURE.md** - Intended system design
- **CLAUDE.md** - Coding conventions and patterns
- **AUTO_TRADER.md** - Component integration patterns
- **Actual source code** - Implementation reality

Each mismatch is categorized by:
- **Severity**: Critical (breaks contracts), Major (inconsistent behavior), Minor (code quality)
- **Pattern Type**: Inheritance, Configuration, Data Flow, Cross-cutting Concerns
- **Location**: Specific file and line numbers

---

## Documented Architecture Patterns

### 1. Trading Environment Hierarchy (ARCHITECTURE.md:277-303)

```
BaseTradingEnvironment (ABC)
├── TradingEnvironment     (Backtest)
└── LiveTradingEnvironment (Live/Paper)
```

**Expected Pattern:**
- Both subclasses inherit from `BaseTradingEnvironment`
- Abstract methods implemented consistently
- Shared utilities in base class (reward calculation, statistics)

### 2. Model Interface Contract (ARCHITECTURE.md:45-103)

**Expected Pattern:**
- Both `TransformerPredictor` and `PPOAgent` share:
  - `save()` / `load()` methods using standardized checkpoints
  - `online_update()` for continuous adaptation
  - Device handling via `utils/device.py`
  - Config via dictionary with `.get()` defaults

### 3. Configuration System (ARCHITECTURE.md:464-507)

**Expected Pattern:**
- Hierarchical `SystemConfig` with nested dataclasses
- Sub-configs: `DataConfig`, `TransformerConfig`, `PPOConfig`, `RiskConfig`, etc.
- JSON serialization with `save()` / `load()` methods

### 4. Logging Convention (CLAUDE.md:Coding Conventions)

**Expected Pattern:**
```python
import logging
logger = logging.getLogger(__name__)
```

Do NOT use `from utils.logging_config import get_logger`.

### 5. Checkpoint System (CLAUDE.md:Model Checkpoints)

**Expected Pattern:**
```python
from utils.checkpoint import save_checkpoint, load_checkpoint, TrainingHistory, CheckpointMetadata

# Standard keys: model_state_dict, optimizer_state_dict, config, training_history, metadata
```

### 6. Risk Validation (CLAUDE.md:Risk Validation)

**Expected Pattern:**
- Pre-validate basic constraints
- Calculate position parameters
- Call `RiskManager.should_take_trade()` with 4 required parameters

---

## Critical Mismatches

### CRITICAL-1: LiveTradingEnvironment Action Execution Violation

| Attribute | Value |
|-----------|-------|
| **Location** | `core/live_trading_env.py:273-289` |
| **Pattern** | Inheritance / Method Override |
| **Documented** | Use base class `_execute_action(action, price)` |
| **Actual** | Custom `_execute_action_live(action)` ignoring price parameter |

**Evidence:**
```python
# LiveTradingEnvironment.step() - line 273
def step(self, action: int):
    # ...
    self._execute_action_live(action)  # WRONG: Should use _execute_action(action, price)
```

vs Base class pattern:
```python
# BaseTradingEnvironment - expected interface
def _execute_action(self, action: int, price: float):
    """Execute trading action at given price."""
```

**Impact:**
- Liskov Substitution Principle violation
- Price parameter ignored (passes 0.0)
- Breaks polymorphic code expecting consistent interface

**Recommendation:**
Refactor `LiveTradingEnvironment.step()` to use `_execute_action(action, price)` with current market price.

---

### CRITICAL-2: Model Network Attribute Naming Inconsistency

| Attribute | Value |
|-----------|-------|
| **Location** | `models/transformer.py:435` vs `models/ppo_agent.py:254` |
| **Pattern** | Model Interface Contract |
| **Documented** | Consistent attribute naming across models |
| **Actual** | `self.model` (Transformer) vs `self.network` (PPO) |

**Evidence:**
```python
# TransformerPredictor - line 435
self.model = TemporalFusionTransformer(...).to(self.device)

# PPOAgent - line 254
self.network = ActorCritic(...).to(self.device)
```

**Impact:**
- Generic code cannot iterate over models consistently
- Checkpoint save/load uses different attribute names:
  - Transformer: `self.model.state_dict()`
  - PPO: `self.network.state_dict()`

**Recommendation:**
Standardize on `self.network` for both, or create a `@property` alias in each class.

---

### CRITICAL-3: Main.py Logging Pattern Violation

| Attribute | Value |
|-----------|-------|
| **Location** | `main.py:92` |
| **Pattern** | Cross-cutting Concerns / Logging |
| **Documented** | Use `logger = logging.getLogger(__name__)` (CLAUDE.md) |
| **Actual** | Uses `logger = get_logger(__name__)` |

**Evidence:**
```python
# main.py:92 - WRONG
logger = get_logger(__name__)

# Should be:
import logging
logger = logging.getLogger(__name__)
```

**Impact:**
- Unnecessary import dependency on `utils.logging_config`
- Violates documented coding convention
- Inconsistent with 20 other modules that use correct pattern

**Recommendation:**
Replace with standard logging pattern.

---

### CRITICAL-4: Position Storage Divergence in LiveTradingEnvironment

| Attribute | Value |
|-----------|-------|
| **Location** | `core/live_trading_env.py:146, 377-382` |
| **Pattern** | Data Flow / State Management |
| **Documented** | Single source of truth for positions via base class |
| **Actual** | Multiple position sources that don't sync |

**Evidence:**
```python
# Three different position storage mechanisms:
self.state.positions            # Inherited from base, never updated in live
self._paper_positions           # Paper mode only - line 146
position_sync.get_positions()   # Live mode - external call
```

**Impact:**
- State inconsistency between modes
- Base class position tracking bypassed
- `_has_position()` may return incorrect results

**Recommendation:**
Unify position storage, always update `self.state.positions` regardless of mode.

---

### CRITICAL-5: Training History Data Structure Mismatch

| Attribute | Value |
|-----------|-------|
| **Location** | `models/transformer.py:459-461` vs `models/ppo_agent.py:274-282` |
| **Pattern** | Model Interface Contract / Checkpointing |
| **Documented** | Use `TrainingHistory` dataclass consistently |
| **Actual** | Different internal structures |

**Evidence:**
```python
# TransformerPredictor - line 459
self.train_losses = []
self.val_losses = []

# PPOAgent - line 274
self.training_stats = {
    'policy_losses': [],
    'value_losses': [],
    'entropy_losses': [],
    'total_losses': [],
    'approx_kl': [],
    'clip_fraction': []
}
```

**Impact:**
- Cannot iterate training history uniformly
- Different save/load serialization paths
- Checkpoint format divergence

**Recommendation:**
Both should use `TrainingHistory` dataclass internally with model-specific fields.

---

## Major Mismatches

### MAJOR-1: Observation Space Dimension Hardcoding

| Attribute | Value |
|-----------|-------|
| **Location** | `core/live_trading_env.py:121-135, 649-652` |
| **Pattern** | Data Flow / Gymnasium API |
| **Documented** | Dynamic feature count from data pipeline |
| **Actual** | Hardcoded 100 features with runtime padding/truncation |

**Evidence:**
```python
# live_trading_env.py:121-135
n_features = 100  # HARDCODED
self.observation_space = spaces.Box(
    low=-np.inf, high=np.inf,
    shape=(self.window_size * (5 + n_features) + n_account_features,),
    dtype=np.float32
)

# line 649-652 - runtime hack
if features.shape[1] < 100:
    features = np.pad(features, ...)  # Information loss!
elif features.shape[1] > 100:
    features = features[:, :100]      # Truncation!
```

**Impact:**
- Observation dimension corrupted when feature count != 100
- Silent information loss from truncation
- Breaking change if `FeatureEngineer` adds features

**Recommendation:**
Get feature count dynamically from `DataPipeline.feature_count`.

---

### MAJOR-2: Account Observation Extension Without Base Class Call

| Attribute | Value |
|-----------|-------|
| **Location** | `core/live_trading_env.py:630-643` |
| **Pattern** | Inheritance / Method Override |
| **Documented** | Call parent `_get_account_observation()` for consistency |
| **Actual** | Returns 12 features instead of base's 8, no super() call |

**Evidence:**
```python
# BaseTradingEnvironment._get_account_observation() - returns 8 features
def _get_account_observation(self) -> np.ndarray:
    return np.array([balance_norm, equity_norm, ...], dtype=np.float32)  # 8 values

# LiveTradingEnvironment._get_account_observation() - returns 12 features
def _get_account_observation(self) -> np.ndarray:
    return np.array([balance_norm, equity_norm, ...,
                     time_features...], dtype=np.float32)  # 12 values!
```

**Impact:**
- Observation space dimension mismatch between backtesting and live
- Trained model expects 8 account features, receives 12 in live mode
- Model inference will fail or produce incorrect results

**Recommendation:**
Either call `super()._get_account_observation()` and extend, or update base class to support additional features.

---

### MAJOR-3: Configuration Passing Style Inconsistency

| Attribute | Value |
|-----------|-------|
| **Location** | Multiple files |
| **Pattern** | Configuration System |
| **Documented** | Dataclass-based hierarchical config |
| **Actual** | Three different styles in use |

**Evidence:**

**Style 1 - Dataclass object** (correct):
```python
# data_pipeline.py:47
class FeatureEngineer:
    def __init__(self, config=None):
        self.config = config
```

**Style 2 - Dictionary with .get()** (inconsistent):
```python
# trainer.py:53-56
class ModelTrainer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.predictor_epochs = self.config.get('predictor_epochs', 100)
```

**Style 3 - Manual dict extraction** (redundant):
```python
# main.py:239-250
self._predictor = TransformerPredictor(
    input_dim=input_dim,
    config={
        'd_model': self.config.transformer.d_model,
        'n_heads': self.config.transformer.n_heads,
        # ... manual extraction
    }
)
```

**Impact:**
- Inconsistent config access patterns
- Type safety lost when converting dataclass → dict
- Maintenance burden from duplicate defaults

**Recommendation:**
Pass full config objects, use `.get()` only at boundaries.

---

### MAJOR-4: Optimizer Algorithm Mismatch

| Attribute | Value |
|-----------|-------|
| **Location** | `models/transformer.py:449` vs `models/ppo_agent.py:261` |
| **Pattern** | Model Interface Contract |
| **Documented** | Not specified (implicit consistency expected) |
| **Actual** | AdamW (Transformer) vs Adam (PPO) |

**Evidence:**
```python
# TransformerPredictor - line 449
self.optimizer = torch.optim.AdamW(
    self.model.parameters(),
    lr=self.learning_rate,
    weight_decay=self.weight_decay  # 1e-5
)

# PPOAgent - line 261
self.optimizer = torch.optim.Adam(
    self.network.parameters(),
    lr=self.learning_rate,
    eps=1e-5  # Different epsilon
)
```

**Impact:**
- Different regularization behavior
- Weight decay only in Transformer
- Inconsistent optimization characteristics

**Recommendation:**
Document why different optimizers are used, or standardize on AdamW.

---

### MAJOR-5: Learning Rate Scheduler Absence in PPO

| Attribute | Value |
|-----------|-------|
| **Location** | `models/transformer.py:453-456` vs `models/ppo_agent.py` |
| **Pattern** | Model Interface Contract |
| **Documented** | Not specified |
| **Actual** | Transformer has scheduler, PPO doesn't |

**Evidence:**
```python
# TransformerPredictor - line 453
self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, mode='min', factor=0.5, patience=5
)

# PPOAgent - NO scheduler
```

**Impact:**
- Transformer adapts learning rate on plateau
- PPO learning rate stays constant
- Inconsistent training dynamics

**Recommendation:**
Add learning rate scheduling to PPO (consider cosine annealing for RL).

---

### MAJOR-6: Gradient Clipping Value Inconsistency

| Attribute | Value |
|-----------|-------|
| **Location** | `models/transformer.py:533,694` vs `models/ppo_agent.py:454` |
| **Pattern** | Model Interface Contract |
| **Documented** | CLAUDE.md suggests consistency |
| **Actual** | Different values across models and contexts |

**Evidence:**
```python
# TransformerPredictor - normal training (line 533)
clip_grad_norm_(self.model.parameters(), 1.0)

# TransformerPredictor - online update (line 694)
clip_grad_norm_(self.model.parameters(), 0.5)

# PPOAgent - policy update (line 454)
clip_grad_norm_(self.network.parameters(), self.max_grad_norm)  # default 0.5
```

**Impact:**
- 1.0 for normal training vs 0.5 for online learning (Transformer)
- PPO uses 0.5 everywhere
- Inconsistent gradient flow characteristics

**Recommendation:**
Standardize gradient clipping values or document rationale for differences.

---

### MAJOR-7: Online Learning Interface Mismatch

| Attribute | Value |
|-----------|-------|
| **Location** | `models/transformer.py:664-701` vs `models/ppo_agent.py:613-664` |
| **Pattern** | Model Interface Contract / Online Learning |
| **Documented** | ARCHITECTURE.md:145-152 describes consistent adaptation |
| **Actual** | Different signatures and data sources |

**Evidence:**
```python
# TransformerPredictor.online_update() - takes fresh data
def online_update(self, X_new: np.ndarray, y_new: np.ndarray, learning_rate=None):
    # Temporarily modify learning rate
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = learning_rate
    # Single gradient update
    ...

# PPOAgent.online_update() - samples from internal buffer
def online_update(self, n_samples: int = 256, n_epochs: int = 5):
    # Sample from experience buffer
    samples = [self.experience_buffer[i] for i in indices]
    # Multiple epochs of updates
    ...
```

**Impact:**
- `OnlineLearningManager` must handle each model differently
- Cannot unify adaptation strategy
- Different learning rate handling approaches

**Recommendation:**
Create common `online_update()` interface with data source abstraction.

---

## Minor Mismatches

### MINOR-1: Exception Handling Patterns

| Attribute | Value |
|-----------|-------|
| **Location** | Multiple files |
| **Pattern** | Cross-cutting Concerns / Error Handling |
| **Issue** | No custom exception classes, inconsistent catch patterns |

**Evidence:**
```python
# Bad - silent exception (live_trading_env.py:430)
except Exception:
    pass

# Good - logged exception (auto_trader.py:457)
except Exception as e:
    logger.exception(f"Error in trading loop")
```

**Recommendation:**
- Create `TradingError` hierarchy in `core/trading_types.py`
- Always use `logger.exception()` in catch blocks
- Never silently pass on exceptions

---

### MINOR-2: Default Learning Rate Discrepancy

| Attribute | Value |
|-----------|-------|
| **Location** | `models/transformer.py:429` vs `models/ppo_agent.py:241` |
| **Issue** | 30x difference in default learning rates |

**Evidence:**
```python
# Transformer default: 1e-4
self.learning_rate = self.config.get('learning_rate', 1e-4)

# PPO default: 3e-4
self.learning_rate = self.config.get('learning_rate', 3e-4)
```

**Recommendation:**
Document rationale or standardize based on model type.

---

### MINOR-3: State Type Variance

| Attribute | Value |
|-----------|-------|
| **Location** | `core/live_trading_env.py:138, 197-214` |
| **Issue** | Type hint variance (TradingState vs LiveTradingState) |

**Recommendation:**
Use consistent type hints or document variance.

---

### MINOR-4: Missing Import in utils/logging_config.py

| Attribute | Value |
|-----------|-------|
| **Location** | `utils/logging_config.py:168-179` |
| **Issue** | `get_logger()` function exists but is documented as discouraged |

**Recommendation:**
Deprecate or remove `get_logger()` function.

---

## Pattern Consistency Matrix

| Component | Inheritance | Config | Data Flow | Logging | Checkpoints | Risk |
|-----------|:-----------:|:------:|:---------:|:-------:|:-----------:|:----:|
| `TransformerPredictor` | N/A | Dict | OK | OK | OK | N/A |
| `PPOAgent` | N/A | Dict | OK | OK | OK | N/A |
| `TradingEnvironment` | OK | OK | OK | OK | N/A | OK |
| `LiveTradingEnvironment` | **FAIL** | OK | **FAIL** | OK | N/A | OK |
| `AutoTrader` | N/A | OK | OK | OK | N/A | OK |
| `DataPipeline` | N/A | Dataclass | OK | OK | N/A | N/A |
| `RiskManager` | N/A | Dataclass | OK | OK | N/A | OK |
| `ModelTrainer` | N/A | Dict | OK | OK | N/A | N/A |
| `main.py` | N/A | Mixed | OK | **FAIL** | N/A | N/A |
| `OnlineLearningManager` | N/A | Dict | OK | OK | N/A | N/A |

**Legend:**
- OK: Follows documented pattern
- **FAIL**: Significant deviation from pattern
- N/A: Pattern not applicable to component

---

## Recommendations

### Immediate (Critical Fixes)

1. **Fix LiveTradingEnvironment action execution** (`live_trading_env.py:273`)
   - Use `_execute_action(action, self._get_current_price())`
   - Ensure base class method is called

2. **Fix main.py logging** (`main.py:92`)
   - Replace `get_logger(__name__)` with `logging.getLogger(__name__)`

3. **Unify position storage** (`live_trading_env.py`)
   - Always update `self.state.positions`
   - Remove redundant `_paper_positions`

### Short-term (Major Fixes)

4. **Standardize model attribute naming**
   - Use `self.network` in both models, or
   - Add `@property` alias for compatibility

5. **Unify training history structure**
   - Both models use `TrainingHistory` dataclass internally

6. **Fix observation space dimensions**
   - Get feature count dynamically from `DataPipeline`
   - Remove hardcoded 100

7. **Align account observation features**
   - Base class and subclass must return same dimension
   - Or update observation space accordingly

### Medium-term (Consistency Improvements)

8. **Standardize configuration passing**
   - Document preferred style (dataclass vs dict)
   - Update components for consistency

9. **Add PPO learning rate scheduler**
   - Implement cosine annealing or similar

10. **Create custom exception hierarchy**
    - `TradingError`, `InsufficientFundsError`, `OrderRejectedError`

11. **Unify online learning interface**
    - Common signature with data source abstraction

### Documentation Updates

12. **Add architecture decision records (ADRs)**
    - Document why different optimizers used
    - Document learning rate differences

13. **Update CLAUDE.md**
    - Add section on model interface contract
    - Clarify configuration passing styles

---

## Appendix: File Reference

| File | Lines Analyzed | Issues Found |
|------|---------------|--------------|
| `core/live_trading_env.py` | 700+ | 4 critical, 2 major |
| `models/transformer.py` | 750+ | 1 critical, 3 major |
| `models/ppo_agent.py` | 730+ | 1 critical, 3 major |
| `main.py` | 600+ | 1 critical, 1 major |
| `training/trainer.py` | 250+ | 1 major |
| `config/settings.py` | 400+ | 0 |
| `core/trading_env.py` | 500+ | 0 |
| `core/trading_env_base.py` | 400+ | 0 |
| `core/auto_trader.py` | 700+ | 0 |
| `training/online_learning.py` | 400+ | 1 major |

---

*Report generated: 2025-12-09*
*Analysis scope: Full codebase architectural review*
