# Before Implementing a New Feature

Before writing code for this feature, complete this checklist:

## 1. Search Existing Utilities
- [ ] Check `utils/` for existing solution
- [ ] Check `core/trading_types.py` for existing types
- [ ] Search for similar patterns in codebase

## 2. Check Documentation
- [ ] Read relevant ADRs in `docs/decisions/`
- [ ] Check module CLAUDE.md in target directory
- [ ] Review `ARCHITECTURE.md` for component relationships

## 3. Verify Patterns
- [ ] Logging: Use `logging.getLogger(__name__)`
- [ ] Errors: Use TradingError hierarchy when appropriate
- [ ] Config: Use EnvConfig or SystemConfig patterns
- [ ] Checkpoints: Use `utils/checkpoint.py` for model saves

## 4. Plan Implementation
- [ ] List files that will be modified
- [ ] Identify tests that need updating
- [ ] Note any ADRs that should be created

## Feature: $ARGUMENTS
