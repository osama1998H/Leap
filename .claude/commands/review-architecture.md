# Architecture Review Before Changes

## 1. Read Relevant ADRs
Check `docs/decisions/` for decisions related to:
- [ ] Component being modified
- [ ] Patterns being used
- [ ] Integration points affected

## 2. Check Module Context
Read the CLAUDE.md in the module you're modifying:
- `core/CLAUDE.md` - Core components
- `models/CLAUDE.md` - AI models
- `utils/CLAUDE.md` - Utilities

## 3. Verify Pattern Consistency
- [ ] Logging: `logging.getLogger(__name__)`
- [ ] Errors: TradingError hierarchy
- [ ] Config: EnvConfig/SystemConfig patterns
- [ ] Checkpoints: `utils/checkpoint.py`
- [ ] PnL: `utils/pnl_calculator.py`
- [ ] Position Sizing: `utils/position_sizing.py` or RiskManager

## 4. Impact Analysis
- [ ] List files that import the code being changed
- [ ] Identify tests that cover this code
- [ ] Check for breaking changes

## Changes: $ARGUMENTS
