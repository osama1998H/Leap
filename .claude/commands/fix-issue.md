# Systematic Debugging Workflow

## 1. Understand the Issue
- [ ] Document expected vs actual behavior
- [ ] Identify steps to reproduce
- [ ] Check logs for error messages

## 2. Locate the Problem
- [ ] Grep for error messages in codebase
- [ ] Trace call stack from entry point (main.py)
- [ ] Check relevant module's CLAUDE.md for gotchas

## 3. Check Existing Patterns
Before fixing, verify the fix follows conventions:
- [ ] Similar fixes in git history?
- [ ] Uses existing utilities (pnl_calculator, position_sizing)?
- [ ] Error handling uses TradingError hierarchy?

## 4. Implement Fix
- [ ] Make minimal change to fix root cause
- [ ] Add/update tests for the fix
- [ ] Check for similar issues elsewhere

## 5. Verify
- [ ] Run related tests: `python -m pytest tests/test_<module>.py -v`
- [ ] Check for regressions
- [ ] Update documentation if behavior changed

## Issue: $ARGUMENTS
