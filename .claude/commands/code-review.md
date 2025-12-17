# Code Review Against Conventions

## 1. Pattern Consistency
- [ ] Uses `logging.getLogger(__name__)` (NOT custom get_logger)
- [ ] Uses TradingError hierarchy for trading exceptions
- [ ] Uses `EnvConfig` for environment configuration
- [ ] Uses `utils/checkpoint.py` for model save/load
- [ ] Uses `utils/pnl_calculator.py` for PnL calculations
- [ ] Uses `utils/position_sizing.py` or RiskManager for sizing

## 2. Architecture Alignment
- [ ] No ADR violations (check `docs/decisions/`)
- [ ] No circular imports introduced
- [ ] No duplication of existing utilities
- [ ] Follows module responsibilities (core/, models/, utils/)

## 3. Code Quality
- [ ] Type hints present
- [ ] Docstrings for public methods
- [ ] No hardcoded values that should be config
- [ ] Error handling appropriate

## 4. Testing
- [ ] Tests added/updated for changes
- [ ] Tests are meaningful (not just coverage)
- [ ] Edge cases considered

## Review: $ARGUMENTS
