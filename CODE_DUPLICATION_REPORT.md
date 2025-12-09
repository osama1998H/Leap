# Code Duplication and Inconsistency Report

## Overview

This report identifies areas in the Leap Trading System codebase where similar functionality is implemented differently across files. Each section documents the inconsistencies with file paths, code snippets, and recommendations for standardization.

**Report Date:** December 2024
**Codebase:** Leap AI Trading System

---

## Table of Contents

1. [Metrics Calculation Duplication](#1-metrics-calculation-duplication)
2. [Position Sizing Implementations](#2-position-sizing-implementations)
3. [Drawdown Calculation Patterns](#3-drawdown-calculation-patterns)
4. [Win Rate and Profit Factor Calculations](#4-win-rate-and-profit-factor-calculations)
5. [Equity Curve Processing](#5-equity-curve-processing)
6. [Model Save/Load Patterns](#6-model-saveload-patterns)
7. [Logging Configuration](#7-logging-configuration)
8. [Summary of Recommendations](#8-summary-of-recommendations)

---

## 1. Metrics Calculation Duplication

### Issue
**MEDIUM Priority** - Sharpe ratio, Sortino ratio, and other risk metrics are calculated in multiple locations with slightly different implementations.

### Location 1: `evaluation/backtester.py:489-509`

```python
def _calculate_results(self) -> BacktestResult:
    # ...
    # Risk metrics
    volatility = np.std(returns) * np.sqrt(periods_per_year) if len(returns) > 1 else 0
    downside_returns = returns[returns < 0]
    downside_vol = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 1 else 0.001

    # Sharpe and Sortino
    risk_free_rate = 0.02  # 2% annual
    excess_return = annualized_return - risk_free_rate
    sharpe = excess_return / volatility if volatility > 0 else 0
    sortino = excess_return / downside_vol if downside_vol > 0 else 0
```

### Location 2: `evaluation/metrics.py:151-175`

```python
def sharpe_ratio(self, returns: np.ndarray) -> float:
    """Calculate Sharpe ratio."""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - self.risk_free_rate / self.periods_per_year
    vol = np.std(returns)

    if vol == 0:
        return 0.0

    return np.mean(excess_returns) / vol * np.sqrt(self.periods_per_year)

def sortino_ratio(self, returns: np.ndarray, target: float = 0.0) -> float:
    """Calculate Sortino ratio."""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - self.risk_free_rate / self.periods_per_year
    downside_vol = self.downside_volatility(returns, target) / np.sqrt(self.periods_per_year)

    if downside_vol == 0:
        return 0.0

    return np.mean(excess_returns) * np.sqrt(self.periods_per_year) / (downside_vol * np.sqrt(self.periods_per_year))
```

### Differences
| Aspect | backtester.py | metrics.py |
|--------|--------------|------------|
| Risk-free rate | Hardcoded `0.02` | Configurable via class attribute |
| Periods per year | Hardcoded `252 * 24` | Configurable via class attribute |
| Excess returns | `annualized_return - risk_free_rate` | `returns - risk_free_rate / periods_per_year` (per-period) |
| Sortino formula | Simplified | Full formula with target parameter |

### Recommendation
**Standardize on `evaluation/metrics.py:MetricsCalculator`**

The `MetricsCalculator` class provides:
- Configurable risk-free rate and periods per year
- More complete Sortino implementation with target parameter
- Consistent methodology across all metrics

**Suggested Refactor:**
```python
# In backtester.py:_calculate_results()
# Replace inline calculations with:
from evaluation.metrics import MetricsCalculator

def _calculate_results(self) -> BacktestResult:
    metrics_calc = MetricsCalculator(
        risk_free_rate=0.02,
        periods_per_year=252 * 24
    )

    equity = np.array(self.equity_curve)
    returns = np.diff(equity) / equity[:-1]

    sharpe = metrics_calc.sharpe_ratio(returns)
    sortino = metrics_calc.sortino_ratio(returns)
    volatility = metrics_calc.volatility(returns)
    # ... etc
```

---

## 2. Position Sizing Implementations

### Issue
**LOW Priority** - Position sizing has both a centralized `RiskManager` implementation and inline fallback calculations.

### Location 1: `core/risk_manager.py:189-226`

```python
def calculate_position_size(
    self,
    entry_price: float,
    stop_loss_price: float,
    volatility: Optional[float] = None
) -> float:
    """Calculate optimal position size based on configured method."""
    if not self.state.is_trading_allowed:
        return 0.0

    risk_per_unit = abs(entry_price - stop_loss_price)

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

### Location 2: `evaluation/backtester.py:159-163` (Inline fallback)

```python
else:
    # Fallback: inline position sizing based on risk
    risk_amount = self.balance * self.risk_per_trade
    size = risk_amount / (stop_loss_pips * self.pip_value * entry_price)
```

### Location 3: `core/trading_env.py:206-209`

```python
def _open_position(self, position_type: str, price: float):
    """Open a new position."""
    # Calculate position size based on risk
    position_value = self.state.balance * self.max_position_size
    size = position_value / price
```

### Differences
| Location | Method | Risk Parameter |
|----------|--------|----------------|
| RiskManager | Multiple methods (kelly, percent, volatility, fixed) | `sizing.percent_risk` |
| Backtester fallback | Risk-based calculation | `self.risk_per_trade` |
| TradingEnvironment | Simple percent of balance | `self.max_position_size` |

### Recommendation
**Already partially standardized** - The `Backtester` correctly delegates to `RiskManager` when available and only falls back to inline calculation when no `RiskManager` is configured (see `backtester.py:132-172`).

**Additional Improvement:**
Add `RiskManager` integration to `TradingEnvironment._open_position()`:

```python
def _open_position(self, position_type: str, price: float):
    if self.risk_manager is not None:
        # Calculate stop loss first
        stop_loss = price * (1 - self.stop_loss_pct) if position_type == 'long' else price * (1 + self.stop_loss_pct)
        size = self.risk_manager.calculate_position_size(price, stop_loss)
    else:
        # Existing fallback
        position_value = self.state.balance * self.max_position_size
        size = position_value / price
```

---

## 3. Drawdown Calculation Patterns

### Issue
**LOW Priority** - Maximum drawdown is calculated in three different locations with identical logic but could be consolidated.

### Location 1: `evaluation/backtester.py:512-514`

```python
# Drawdown
peak = np.maximum.accumulate(equity)
drawdown = (peak - equity) / peak
max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
```

### Location 2: `evaluation/metrics.py:115-121`

```python
def max_drawdown(self, equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    return np.max(drawdown)
```

### Location 3: `evaluation/backtester.py:824-827` (MonteCarloSimulator)

```python
for sim in simulated_equity:
    peak = np.maximum.accumulate(sim)
    dd = (peak - sim) / peak
    max_drawdowns.append(np.max(dd))
```

### Recommendation
**Standardize on `evaluation/metrics.py:MetricsCalculator.max_drawdown()`**

All three implementations are functionally identical. Import and use the `MetricsCalculator` method:

```python
# In MonteCarloSimulator.simulate_from_trades()
from evaluation.metrics import MetricsCalculator
metrics = MetricsCalculator()

for sim in simulated_equity:
    max_drawdowns.append(metrics.max_drawdown(np.array(sim)))
```

---

## 4. Win Rate and Profit Factor Calculations

### Issue
**LOW Priority** - Trade statistics (win rate, profit factor) are calculated in multiple locations.

### Location 1: `evaluation/backtester.py:530-541`

```python
# Trade statistics
trade_pnls = [t.pnl for t in self.closed_trades]
winning_trades = [t for t in self.closed_trades if t.pnl > 0]
losing_trades = [t for t in self.closed_trades if t.pnl <= 0]

n_trades = len(self.closed_trades)
n_winners = len(winning_trades)
n_losers = len(losing_trades)
win_rate = n_winners / n_trades if n_trades > 0 else 0

gross_profit = sum(t.pnl for t in winning_trades)
gross_loss = abs(sum(t.pnl for t in losing_trades))
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
```

### Location 2: `core/trading_types.py:179-196` (TradeStatistics)

```python
@property
def win_rate(self) -> float:
    """Calculate win rate as a ratio (0.0 to 1.0)."""
    if self.total_trades == 0:
        return 0.0
    return self.winning_trades / self.total_trades

@property
def profit_factor(self) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    if self.gross_loss == 0:
        return float('inf') if self.gross_profit > 0 else 0.0
    return self.gross_profit / self.gross_loss
```

### Location 3: `evaluation/metrics.py:268-306`

```python
def trade_metrics(self, trades: List) -> Dict:
    """Calculate trade-level metrics."""
    pnls = [t.pnl for t in trades]
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]

    return {
        'total_trades': len(trades),
        'winning_trades': len(winners),
        'losing_trades': len(losers),
        'win_rate': len(winners) / len(trades) if trades else 0,
        # ...
    }
```

### Differences
| Location | Losing trade definition | Profit factor when no losses |
|----------|------------------------|------------------------------|
| backtester.py | `pnl <= 0` | `float('inf')` |
| TradeStatistics | Has `breakeven_trades` category | `float('inf')` if profit > 0, else `0.0` |
| metrics.py | `pnl <= 0` | Separate `_profit_factor` method |

### Recommendation
**Standardize on `core/trading_types.py:TradeStatistics`**

The `TradeStatistics` dataclass is already designed as a reusable component:

```python
# In backtester.py:_calculate_results()
from core.trading_types import TradeStatistics, Trade

def _calculate_results(self) -> BacktestResult:
    stats = TradeStatistics()
    for trade in self.closed_trades:
        stats.update_from_trade(trade)

    # Use stats.win_rate, stats.profit_factor, etc.
```

---

## 5. Equity Curve Processing

### Issue
**INFO** - Returns calculation from equity curve appears in multiple places with identical logic.

### Location 1: `evaluation/backtester.py:492`

```python
returns = np.diff(equity) / equity[:-1]
```

### Location 2: `evaluation/metrics.py:78-80`

```python
def _calculate_returns(self, equity_curve: np.ndarray) -> np.ndarray:
    """Calculate returns from equity curve."""
    return np.diff(equity_curve) / equity_curve[:-1]
```

### Recommendation
**Already standardized** - The `MetricsCalculator._calculate_returns()` method exists. The inline usage in `backtester.py` could use it but this is a minor inconsistency that doesn't impact functionality.

---

## 6. Model Save/Load Patterns

### Issue
**RESOLVED** - Previously had inconsistent checkpoint formats between TransformerPredictor and PPOAgent. Now standardized.

### Current Implementation: `utils/checkpoint.py`

The codebase now has a centralized checkpoint system with:
- `CHECKPOINT_KEYS` - Standard key constants
- `CheckpointMetadata` - Architecture information
- `TrainingHistory` - Unified training metrics
- `save_checkpoint()` / `load_checkpoint()` - Standardized functions

### Legacy Support
The `load_checkpoint()` function handles backward compatibility:

```python
# Legacy key mappings for backward compatibility
LEGACY_KEY_MAPPINGS = {
    'network_state_dict': 'model_state_dict',  # PPOAgent legacy
    'training_stats': 'training_history',       # PPOAgent legacy
    'train_losses': 'training_history',         # Transformer legacy
    'val_losses': 'training_history',           # Transformer legacy
}
```

### Recommendation
**No action needed** - This is well-implemented. Continue using `utils/checkpoint.py` for all model save/load operations.

---

## 7. Logging Configuration

### Issue
**RESOLVED** - Previously had two logging patterns. Now standardized per CLAUDE.md guidelines.

### Standard Pattern (Used throughout codebase)

```python
import logging
logger = logging.getLogger(__name__)
```

### Files using standard pattern:
- `models/transformer.py:10-11`
- `models/ppo_agent.py:14-15`
- `training/trainer.py:12-13`
- `training/online_learning.py:13-14`
- `core/risk_manager.py:11-12`
- `core/trading_env.py:9-10`
- `evaluation/backtester.py:22-23`
- `evaluation/metrics.py:11-12`
- `utils/checkpoint.py:6,11`
- `utils/mlflow_tracker.py:8,39`

### Recommendation
**Already standardized** - All modules consistently use `import logging; logger = logging.getLogger(__name__)`. The `utils/logging_config.py` module provides the `setup_logging()` function for initial configuration but is not used for creating loggers in individual modules.

---

## 8. Summary of Recommendations

### High Priority (Significant Code Quality Impact)
*None identified - codebase is well-structured*

### Medium Priority (Should Address)

| Issue | Location | Recommendation |
|-------|----------|----------------|
| Metrics calculation duplication | `backtester.py`, `metrics.py` | Refactor `backtester._calculate_results()` to use `MetricsCalculator` |

### Low Priority (Optional Improvements)

| Issue | Location | Recommendation |
|-------|----------|----------------|
| Position sizing in TradingEnv | `trading_env.py:206-209` | Add RiskManager delegation |
| Trade statistics duplication | Multiple | Use `TradeStatistics` dataclass consistently |
| Drawdown calculation | Multiple | Consolidate to single `MetricsCalculator.max_drawdown()` call |

### Already Resolved

| Issue | Status |
|-------|--------|
| Model checkpoint formats | ✅ Standardized in `utils/checkpoint.py` |
| Logging patterns | ✅ Consistent `logging.getLogger(__name__)` |
| Device management | ✅ Centralized in `utils/device.py` |
| Configuration system | ✅ Dataclass-based in `config/settings.py` |
| Trade/Position types | ✅ Consolidated in `core/trading_types.py` |

---

## Conclusion

The Leap Trading System codebase demonstrates good architectural decisions with centralized utilities and consistent patterns. The main area for improvement is consolidating metrics calculations to use the `MetricsCalculator` class consistently, which would reduce code duplication and ensure consistent methodology across backtesting and analysis.

The existing documentation in `CLAUDE.md` provides excellent guidance for maintaining these patterns going forward.
