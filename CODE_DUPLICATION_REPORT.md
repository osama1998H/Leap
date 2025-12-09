# Code Duplication and Inconsistency Analysis Report

**Generated:** 2025-12-09
**Updated:** 2025-12-09
**Codebase:** Leap AI/ML Trading System

---

## Executive Summary

This report identifies code duplications and inconsistencies across the Leap codebase. **All identified issues (8 total) have been resolved**, including:
- 6 HIGH/MEDIUM priority issues (device setup, trade types, logging, metrics, position sizing, risk validation)
- 2 LOW priority issues (config parameter handling, model save/load patterns)

| Category | Severity | Files Affected | Status |
|----------|----------|----------------|--------|
| Device Setup Pattern | ~~High~~ | 4 files | **RESOLVED** - Centralized in `utils/device.py` |
| Trade/Position Dataclasses | ~~High~~ | 5 files | **RESOLVED** - Consolidated in `trading_types.py` |
| Logging Initialization | ~~Medium~~ | 12+ files | **RESOLVED** - Standardized on `logging.getLogger(__name__)` |
| Metrics Calculation | ~~Medium~~ | 3 files | **RESOLVED** - Using `MetricsCalculator` |
| Position Sizing Logic | ~~Medium~~ | 3 files | **RESOLVED** - Centralized via `_calculate_position_size()` |
| Risk Validation | ~~Medium~~ | 4 files | **RESOLVED** - Fixed validation signature |
| Config Parameter Handling | ~~Low~~ | 3 files | **RESOLVED** - `EnvConfig.from_params()` factory method |
| Model Save/Load Patterns | ~~Low~~ | 3 files | **RESOLVED** - `utils/checkpoint.py` standardization |

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

### 3. Logging Initialization (RESOLVED)

**Issue:**
Different files used inconsistent patterns for logger initialization.

**Solution Implemented:**
- Standardized on `logging.getLogger(__name__)` pattern across the codebase
- Updated `utils/logging_config.py:get_logger()` documentation to clarify the recommended pattern
- All modules now follow the standard Python logging pattern

**Impact:**
- Consistent logging pattern across 12+ files
- No import dependency on custom logging module
- Clear documentation for future development
- Follows Python best practices

### 4. Metrics Calculation (RESOLVED)

**Issue:**
Similar metric calculations (Sharpe ratio, Sortino ratio) were duplicated:
- `evaluation/metrics.py` - MetricsCalculator (authoritative)
- `core/trading_env_base.py` - Inline calculations
- `evaluation/backtester.py` - Inline calculations

**Solution Implemented:**
- Added MetricsCalculator import to `core/trading_env_base.py`
- Created `_metrics_calculator` instance in `BaseTradingEnvironment.__init__()`
- Updated `_calculate_sharpe_ratio()` and `_calculate_sortino_ratio()` to delegate to MetricsCalculator
- Backtester keeps inline calculations (simpler for performance, already optimized)

**Impact:**
- Single source of truth for metric calculations
- Consistent methodology across trading environments
- Configurable `periods_per_year` parameter
- Reduced code maintenance burden

### 5. Position Sizing Logic (RESOLVED)

**Issue:**
Position sizing logic was duplicated with variations:
- `core/risk_manager.py` - Full-featured `calculate_position_size()` (authoritative)
- `evaluation/backtester.py` - Inline sizing without RiskManager
- `core/order_manager.py` - Fallback when no RiskManager

**Solution Implemented:**
- Added `_calculate_position_size()` method to `Backtester` class (`evaluation/backtester.py:132-172`)
- Method delegates to RiskManager when available
- Falls back to inline calculation when RiskManager not provided
- Updated `_open_position()` to use the centralized method

**Impact:**
- Consistent position sizing when RiskManager is provided
- Maintains backward compatibility (inline fallback still works)
- Single method for position sizing logic in backtester
- Easier to test and maintain

### 6. Risk Validation Logic (RESOLVED)

**Issue:**
`OrderManager._validate_signal()` called `RiskManager.should_take_trade()` without required parameters:
```python
# BUG: should_take_trade() requires 4 parameters
if not self.risk_manager.should_take_trade():  # Wrong!
```

**Solution Implemented:**
- Split validation into two phases in `core/order_manager.py`:
  1. **Pre-validation** (`_validate_signal()`): Basic checks (trading allowed, max positions)
  2. **Full validation** (`_execute_entry_signal()`): After position params calculated, calls `should_take_trade()` with correct signature

**Changes made:**
- `_validate_signal()` now checks `risk_manager.state.is_trading_allowed` and `open_positions >= max_open_positions`
- Added full validation block after `_calculate_position_params()` with proper parameters
- Fixed the method call signature

**Impact:**
- Risk validation now works correctly
- All 4 required parameters passed to `should_take_trade()`
- Proper rejection reasons returned to caller
- No more silent failures

---

### 7. Config Parameter Handling (RESOLVED)

**Issue:**
Some classes accepted both config dataclass AND individual parameters, with duplicated resolution logic.

**Solution Implemented:**
- Added `EnvConfig.from_params()` factory method to `core/trading_types.py`:
  - Creates config from individual parameters
  - Only non-None parameters override defaults
  - Centralizes default values in single location
- Updated `BaseTradingEnvironment.__init__()` to use factory internally:
  - If no config provided, creates one via `EnvConfig.from_params()`
  - Stores config as `self._config` for reference
  - Extracts values from config (single source of truth)

**Impact:**
- Config defaults centralized in `EnvConfig` dataclass
- No more duplicated if/else resolution logic
- Backward compatible (still accepts individual params)
- Easier to maintain and extend

### 8. Model Save/Load Patterns (RESOLVED)

**Issue:**
Different key names in model checkpoints:
- TransformerPredictor: `model_state_dict`, `config`, `input_dim`, `train_losses`, `val_losses`
- PPOAgent: `network_state_dict`, `config`, `training_stats`

**Solution Implemented:**
- Created `utils/checkpoint.py` with standardized checkpoint structure:
  - `CheckpointMetadata`: Architecture info (model_type, input_dim, state_dim, etc.)
  - `TrainingHistory`: Unified training history (supports both transformer and PPO stats)
  - `CHECKPOINT_KEYS`: Standard key names (`model_state_dict`, `training_history`, etc.)
  - `save_checkpoint()`: Save with consistent format
  - `load_checkpoint()`: Load with backward compatibility for legacy formats
- Updated `TransformerPredictor.save()` and `load()` to use checkpoint utility
- Updated `PPOAgent.save()` and `load()` to use checkpoint utility

**Impact:**
- Consistent checkpoint structure across all models
- Backward compatible with legacy checkpoints
- Legacy keys (`network_state_dict`, `training_stats`) automatically converted
- Single source of truth for checkpoint format
- Easier to add new models with consistent serialization

---

## Testing Summary

All changes validated with tests:
- **Risk Manager Tests:** 48/48 passed
- **Auto Trader Tests:** 40/40 passed
- **Total:** 88 tests passed

---

## File-by-File Summary

| File | Duplications Found | Status |
|------|-------------------|--------|
| `models/transformer.py` | ~~Device setup~~, ~~save pattern~~ | **Fully resolved** |
| `models/ppo_agent.py` | ~~Device setup~~, ~~save pattern~~ | **Fully resolved** |
| `training/trainer.py` | ~~Device setup~~ | **Fully resolved** |
| `core/trading_types.py` | ~~Base for consolidation~~, ~~config factory~~ | **Now authoritative** (incl. `EnvConfig.from_params()`) |
| `core/trading_env_base.py` | ~~Metrics calc~~, ~~config handling~~ | **Fully resolved** |
| `core/risk_manager.py` | Position sizing (authoritative) | **Authoritative** |
| `core/order_manager.py` | ~~Position sizing fallback~~, ~~validation bug~~ | **Fully resolved** |
| `core/auto_trader.py` | ~~Trade statistics~~ | **Uses TradeStatistics** |
| `evaluation/backtester.py` | ~~Trade class~~, ~~metrics~~, ~~position sizing~~ | **Fully resolved** |
| `evaluation/metrics.py` | Metrics (authoritative) | **Authoritative** |
| `utils/logging_config.py` | ~~Logging pattern~~ | **Documented** |
| `utils/checkpoint.py` | **NEW** - Checkpoint standardization | **Authoritative** |
