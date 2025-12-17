# Adding a New Utility or Pattern

## 1. Verify Need
- [ ] Searched `utils/` for existing solution
- [ ] Searched `core/trading_types.py` for existing types
- [ ] Pattern appears 3+ times in codebase (justifies extraction)

## 2. Reference Templates
Use these as examples:
- `utils/pnl_calculator.py` - Simple calculation utility
- `utils/position_sizing.py` - Utility with logging
- `utils/checkpoint.py` - Complex utility with dataclasses

## 3. Implementation Checklist
- [ ] Add docstrings with Examples section
- [ ] Add `__all__` export list
- [ ] Use `logger = logging.getLogger(__name__)`
- [ ] Add unit tests

## 4. Documentation Updates
- [ ] Update CLAUDE.md Centralized Utilities table
- [ ] Update relevant module CLAUDE.md
- [ ] Create ADR in `docs/decisions/` if significant pattern

## New Pattern: $ARGUMENTS
