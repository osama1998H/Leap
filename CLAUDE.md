# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

| Topic | Documentation |
|-------|---------------|
| Project Overview & Features | [README.md](README.md) |
| Directory Structure | [README.md](README.md#project-structure) |
| CLI Commands & Options | [CLI.md](CLI.md) |
| System Architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Configuration System | [ARCHITECTURE.md](ARCHITECTURE.md#configuration-system) |
| Auto-Trader | [docs/AUTO_TRADER.md](docs/AUTO_TRADER.md) |
| Extension Points | [ARCHITECTURE.md](ARCHITECTURE.md#extension-points) |

## Project Overview

Leap is an AI-powered forex trading system combining Transformer-based price prediction with PPO reinforcement learning. Features online learning for continuous market adaptation, walk-forward optimization, and MetaTrader 5 integration for live trading.

## Common Commands

```bash
# Training
python main.py train --symbol EURUSD --epochs 100 --timesteps 100000
python main.py train --symbols EURUSD GBPUSD --multi-timeframe  # Multi-symbol + multi-timeframe

# Backtesting
python main.py backtest --symbol EURUSD --realistic --monte-carlo
python main.py walkforward --symbol EURUSD

# Live Trading (Windows only - requires MT5)
python main.py autotrade --paper --symbol EURUSD

# Testing
python -m pytest tests/ -v
```

> See [CLI.md](CLI.md) for complete command reference with all options.

## Coding Conventions

### Logging

Use the standard Python logging pattern in all modules:

```python
import logging
logger = logging.getLogger(__name__)
```

Do NOT use `from utils.logging_config import get_logger`. The standard pattern has no import dependency and is consistent across the codebase.

### Centralized Utilities

| Utility | Location | Usage |
|---------|----------|-------|
| Device management | `utils/device.py` | Use `resolve_device(device)` for PyTorch device handling |
| Model checkpoints | `utils/checkpoint.py` | Use `save_checkpoint()` / `load_checkpoint()` for consistent model persistence |
| PnL calculations | `utils/pnl_calculator.py` | Use `calculate_pnl()` / `calculate_unrealized_pnl()` for consistent PnL |
| Position sizing | `utils/position_sizing.py` | Use `calculate_risk_based_size()` / `calculate_percentage_size()` for fallback sizing |
| Trade types | `core/trading_types.py` | Use `Trade`, `TradeStatistics` dataclasses |
| Trading exceptions | `core/trading_types.py` | Use `TradingError` hierarchy for trading-related errors |
| Environment config | `core/trading_types.py` | Use `EnvConfig` or `EnvConfig.from_params()` factory |
| Metrics | `evaluation/metrics.py` | Use `MetricsCalculator` for Sharpe, Sortino, etc. |
| Risk management | `core/risk_manager.py` | Delegate to `RiskManager.calculate_position_size()` when available |

### Position Sizing

When calculating position sizes:
1. Use `RiskManager.calculate_position_size()` when a RiskManager is available
2. Fall back to utilities in `utils/position_sizing.py` when no RiskManager is configured:
   - `calculate_risk_based_size()` for risk-based sizing with stop loss
   - `calculate_percentage_size()` for simple percentage-of-balance sizing
   - `apply_position_limits()` for applying leverage and max size constraints
3. See `evaluation/backtester.py:_calculate_position_size()` for the pattern

### Risk Validation

When validating trades:
1. Pre-validate basic constraints (trading allowed, max positions)
2. Calculate position parameters (entry, SL, TP)
3. Call `RiskManager.should_take_trade()` with all 4 required parameters:
   - `entry_price`, `stop_loss_price`, `take_profit_price`, `direction`

### Model Checkpoints

Use the standardized checkpoint system (`utils/checkpoint.py`) for model save/load:

```python
from utils.checkpoint import save_checkpoint, load_checkpoint, TrainingHistory, CheckpointMetadata

# Saving
training_history = TrainingHistory(train_losses=losses, val_losses=val_losses)
metadata = CheckpointMetadata(model_type='transformer', input_dim=128)
save_checkpoint(path, model.state_dict(), optimizer.state_dict(), config, training_history, metadata)

# Loading (with backward compatibility)
checkpoint = load_checkpoint(path, device)
model.load_state_dict(checkpoint['model_state_dict'])
```

**Standard checkpoint keys:** `model_state_dict`, `optimizer_state_dict`, `config`, `training_history`, `metadata`

### Environment Configuration

Use `EnvConfig` for trading environment configuration:

```python
from core.trading_types import EnvConfig

config = EnvConfig(initial_balance=50000.0, leverage=50)
# Or use factory: EnvConfig.from_params(initial_balance=50000.0)
env = TradingEnvironment(data=data, config=config)
```

## Testing

**Test files in `tests/`:**
- `test_cli.py` - CLI and system integration
- `test_integration.py` - End-to-end pipeline
- `test_feature_engineering.py` - Feature computation
- `test_risk_manager.py` - Risk management

**Testing patterns:**
- Mock classes for external dependencies (MT5, data sources)
- Fixture-based test data generation
- Integration tests for full pipelines

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Deep Learning | PyTorch >=2.2.0, NumPy <2.0 (ABI compatibility) |
| Data | Pandas >=2.0, SciPy >=1.11, scikit-learn >=1.3 |
| RL | Gymnasium >=0.29 |
| Trading | MetaTrader5 >=5.0.45 (Windows only, optional) |
| Tracking | MLflow >=2.10, TensorBoard >=2.14 |
| Testing | pytest >=7.4 |

## Search Before Creating

Before implementing any new:
- **Utility function**: Search `utils/` directory first
- **Exception type**: Check `core/trading_types.py` TradingError hierarchy
- **Configuration**: Check `EnvConfig` and `SystemConfig` patterns
- **Test pattern**: Check existing tests in `tests/` for similar patterns

Use slash commands to guide your workflow:
- `/before-feature` - Before implementing new features
- `/review-architecture` - Before making architectural changes
- `/fix-issue` - Systematic debugging workflow
- `/add-pattern` - When adding new utilities
- `/refactor` - Safe refactoring workflow
- `/code-review` - Review against conventions

## Architecture Decision Records

Before making decisions that create new patterns or affect multiple modules, check `docs/decisions/` for existing ADRs. Create a new ADR when:
- Introducing cross-module patterns
- Choosing between significant alternatives
- Establishing conventions that must be followed

## Module-Specific Context

Each major module has its own `CLAUDE.md` with module-specific patterns:
- `core/CLAUDE.md` - Trading system core (EnvConfig, TradingError, environments)
- `models/CLAUDE.md` - AI models (Transformer, PPO)
- `utils/CLAUDE.md` - Centralized utilities

Read the relevant module CLAUDE.md before making changes to that module.

## Living Documentation

After completing work that:
- Introduces new patterns
- Creates reusable utilities
- Changes conventions
- Makes architectural decisions

Update the relevant documentation:
1. Module-level `CLAUDE.md` if patterns change
2. Create ADR in `docs/decisions/` for significant patterns
3. Update this file's Centralized Utilities table if adding utilities

## Important Notes

- **NumPy Version**: Must be <2.0 for PyTorch ABI compatibility
- **MT5 Integration**: Only works on Windows; use paper mode on other platforms
- **Safety**: Always use `--paper` flag when testing `autotrade` command
- **Model Files**: Saved to `saved_models/` with `model_metadata.json` for reloading
- **Logging**: Logs rotate by size (10MB default) with 5 backups
