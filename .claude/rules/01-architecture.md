# Architecture Constraints

This rule defines the hard architectural constraints that must NEVER be violated.
These constraints exist to maintain clean module boundaries and prevent coupling.

---

## Module Layering

Imports flow DOWN the stack, never UP.

```
Entry Point Layer
    └── cli/                     # Commands, parser, system orchestration
            ↓
Business Logic Layer
    ├── core/                    # Trading types, environments, broker, strategies
    ├── models/                  # TransformerPredictor, PPOAgent
    ├── training/                # Trainers, online learning
    └── evaluation/              # Backtester, metrics
            ↓
Infrastructure Layer
    ├── utils/                   # Stateless utilities
    └── config/                  # Configuration dataclasses
```

---

## Forbidden Import Patterns

These patterns create coupling and must be avoided:

| From | To | Why Forbidden |
|------|----|--------------|
| `utils/` | `core/`, `models/`, `training/` | Utils must be dependency-free |
| `config/` | `core/`, `models/`, `training/` | Config defines data, not behavior |
| `core/` | `cli/` | Core shouldn't know about CLI |
| `models/` | `cli/` | Models shouldn't know about CLI |
| Any module | Circular import | Creates runtime errors |

### Handling Optional Dependencies

When a module needs a type hint from a higher layer:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.risk_manager import RiskManager  # Only for type hints
```

This pattern is used in: `AutoTrader`, `OrderManager`, `Backtester`

---

## Module Responsibilities

### `core/` - Trading System Core
- Trading types and dataclasses (`Trade`, `EnvConfig`, `TradingError`)
- Trading environments (`BaseTradingEnvironment`, `TradingEnvironment`, `LiveTradingEnvironment`)
- Broker abstraction (`BrokerGateway` Protocol, `MT5BrokerGateway`, `PaperBrokerGateway`)
- Trading strategies (`TradingStrategy` ABC, `CombinedPredictorAgentStrategy`)
- Risk management (`RiskManager`)
- Order execution (`OrderManager`, `PositionSynchronizer`)
- Data pipeline and feature engineering

### `models/` - AI Models
- `TransformerPredictor`: Price prediction with quantile outputs
- `PPOAgent`: Reinforcement learning for trading decisions
- NO trading logic - models are pure ML

### `utils/` - Stateless Utilities
- `device.py`: PyTorch device management
- `checkpoint.py`: Model persistence
- `pnl_calculator.py`: PnL calculations
- `position_sizing.py`: Position size calculations
- `logging_config.py`: Logging setup
- `mlflow_tracker.py`: Experiment tracking
- `data_saver.py`: Data persistence

### `config/` - Configuration
- Dataclass definitions for all configs
- Modular config loaders
- JSON serialization support

### `cli/` - Command Line Interface
- `LeapTradingSystem`: Main orchestrator (lazy-loads components)
- Command modules: `train`, `backtest`, `walkforward`, `autotrade`, `adapt`
- Parser and argument handling

### `training/` - Model Training
- `ModelTrainer`: Training orchestration
- `OnlineLearningManager`: Continuous adaptation
- `AdaptiveTrainer`: Combined training

### `evaluation/` - Performance Evaluation
- `Backtester`: Historical simulation
- `WalkForwardOptimizer`: Rolling validation
- `MonteCarloSimulator`: Risk analysis
- `MetricsCalculator`: Performance metrics

---

## Key Invariants (NEVER BREAK)

These invariants are critical for system consistency:

### 1. Broker Abstraction
All broker access MUST go through `BrokerGateway` Protocol.
```python
from core.broker_interface import BrokerGateway, create_broker
broker: BrokerGateway = create_broker('paper', initial_balance=10000.0)
```

### 2. Checkpoints
All model persistence MUST use `utils/checkpoint.py`.
```python
from utils.checkpoint import save_checkpoint, load_checkpoint
```

### 3. PnL Calculations
All PnL MUST use `utils/pnl_calculator.py`.
```python
from utils.pnl_calculator import calculate_pnl, calculate_unrealized_pnl
```

### 4. Position Sizing
Position sizing through `RiskManager` when available, fallback to `utils/position_sizing.py`.
```python
if risk_manager:
    size = risk_manager.calculate_position_size(...)
else:
    from utils.position_sizing import calculate_risk_based_size
    size = calculate_risk_based_size(...)
```

### 5. Error Handling
All trading errors MUST use `TradingError` hierarchy.
```python
from core.trading_types import TradingError, InsufficientFundsError
```

### 6. Logging
All modules MUST use standard logging pattern.
```python
import logging
logger = logging.getLogger(__name__)
```
NOT: `from utils.logging_config import get_logger`

---

## Multi-Module Changes

### If a change touches more than 3 modules:

1. **STOP** - Do not proceed with implementation
2. **Propose a refactor plan** - Document what needs to change and why
3. **Get user approval** - Large changes need explicit sign-off
4. **Consider an ADR** - Document the architectural decision

### Warning Signs of Scope Creep
- "While I'm here, I should also..."
- "This would be cleaner if I also changed..."
- "I noticed this bug in another file..."

When you see these patterns, create GitHub issues instead of expanding scope.

---

## Extension Points

When adding new components:

| Type | Location | Pattern to Follow |
|------|----------|-------------------|
| New Model | `models/` | Follow `TransformerPredictor` structure |
| New Strategy | `core/strategy.py` | Extend `TradingStrategy` ABC |
| New Metric | `evaluation/metrics.py` | Add to `MetricsCalculator` |
| New Feature | `core/feature_engineer.py` | Add computation method |
| New Command | `cli/commands/` | Follow handler signature pattern |
| New Utility | `utils/` | Follow existing utility patterns |
