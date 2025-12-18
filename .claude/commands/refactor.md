# Safe Refactoring Workflow

## 1. Pre-Refactor Checklist
- [ ] Run all tests: `python -m pytest tests/ -v`
- [ ] Commit any uncommitted changes

## 2. Identify Impact
- [ ] List all files that import the code being refactored
- [ ] List all tests that cover this code
- [ ] Check git blame for recent changes

## 3. Refactor Strategy
Principles:
- Keep existing public interfaces unless explicitly changing
- Use deprecation warnings for breaking changes
- Update documentation alongside code

## 4. Post-Refactor Verification
- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] No new linting errors
- [ ] Documentation updated
- [ ] CLAUDE.md updated if patterns changed
- [ ] ADR created if architectural change

## Refactoring: $ARGUMENTS
