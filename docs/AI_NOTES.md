# AI Implementation Notes

This file tracks implementation state for long-running or complex tasks.
Updated by Claude during work to prevent context loss across sessions.

**Purpose:**
- Preserve decisions and context across conversation limits
- Track out-of-scope issues discovered
- Document what was changed and why
- Maintain continuity for follow-up sessions

---

## Current Task

<!-- Updated when starting new work -->

- **Task**: Enhanced Context Engineering Implementation
- **Branch**: `feature/enhanced-context-engineering`
- **Started**: 2025-12-18
- **Status**: Complete - Ready for commit

---

## Key Decisions Made

<!-- Decisions that affect implementation -->

1. **Modular rules structure**: Using `.claude/rules/` with path-scoped rules for module-specific constraints
2. **YAML frontmatter**: Path-scoped rules use `paths:` frontmatter for automatic loading
3. **Bug tracking via GitHub**: Out-of-scope bugs create GitHub issues with `discovered-by-agent` label
4. **AI_NOTES.md committed**: This file is tracked in git for team visibility
5. **@imports in CLAUDE.md**: Importing ARCHITECTURE.md and ADR index

---

## Files Changed

<!-- Track what's been modified -->

| File | Change Type | Status |
|------|-------------|--------|
| `.claude/rules/00-operating-procedure.md` | Created | Complete |
| `.claude/rules/01-architecture.md` | Created | Complete |
| `.claude/rules/02-git-workflow.md` | Created | Complete |
| `.claude/rules/03-out-of-scope-bugs.md` | Created | Complete |
| `.claude/rules/models.md` | Created | Complete |
| `.claude/rules/core.md` | Created | Complete |
| `.claude/rules/cli.md` | Created | Complete |
| `.claude/rules/testing.md` | Created | Complete |
| `.claude/rules/config.md` | Created | Complete |
| `.claude/commands/safe-change.md` | Created | Complete |
| `.claude/commands/create-bug-issue.md` | Created | Complete |
| `.claude/commands/start-task.md` | Created | Complete |
| `docs/AI_NOTES.md` | Created | Complete |
| `CLAUDE.md` | Updated | Complete |

---

## Invariants Discovered

<!-- Important constraints found during exploration -->

- Module layering: `cli/` → `core/`, `models/`, `training/`, `evaluation/` → `utils/`, `config/`
- Broker access must go through `BrokerGateway` Protocol
- All checkpoints through `utils/checkpoint.py`
- All PnL through `utils/pnl_calculator.py`
- Logging uses `logging.getLogger(__name__)` (NOT custom get_logger)
- 13 existing ADRs in `docs/decisions/`

---

## Out-of-Scope Issues Found

<!-- Bugs/issues to address separately -->

| Issue # | Description | File | Priority |
|---------|-------------|------|----------|
| - | None found during this task | - | - |

---

## Test Results

<!-- Last test run results -->

```
Command: (not run yet)
Status:
Failures:
```

---

## Next Steps

<!-- What needs to happen next -->

1. Update CLAUDE.md with @imports section
2. Verify all rules load correctly in new session
3. Commit all changes
4. Test the workflow with a sample task

---

## Session History

<!-- Brief summaries of past sessions -->

### 2025-12-18 - Initial Implementation
- Task: Implement enhanced context engineering
- Outcome: Created 9 rules, 3 commands, AI_NOTES template
- Issues created: None
- Notes: Based on Anthropic's context engineering best practices
