# Start New Task

Task: $ARGUMENTS

Use this command at the beginning of any new development task.

---

## Git Setup

### 1. Ensure on Main Branch

```bash
git checkout main
git pull origin main
```

Verify you're on main and up to date before creating a new branch.

### 2. Create Feature Branch

```bash
git checkout -b <type>/<task-slug>
```

**Branch types:**

| Type | Use For |
|------|---------|
| `feature/` | New features |
| `fix/` | Bug fixes |
| `refactor/` | Code refactoring |
| `docs/` | Documentation only |
| `test/` | Test improvements |

**Example:**
---

## Explore Phase

Before writing any code:

### 3. Read Relevant Module CLAUDE.md Files

Based on what you'll be modifying:

- [ ] Root `CLAUDE.md` - Always read first
- [ ] `core/CLAUDE.md` - If touching core/
- [ ] `models/CLAUDE.md` - If touching models/
- [ ] `utils/CLAUDE.md` - If touching utils/
- [ ] `cli/CLAUDE.md` - If touching cli/

### 4. Search for Existing Patterns

Before implementing anything:

- [ ] Search `utils/` for existing utilities
- [ ] Search `core/trading_types.py` for existing types
- [ ] Search for similar implementations in codebase
- [ ] Check if pattern already exists

### 5. List Key Files to Understand

Identify:
- Files you'll modify
- Files with patterns to follow
- Files that might be affected by changes

### 6. Check Relevant ADRs

Look in `docs/decisions/` for:
- Existing decisions that apply to this task
- Patterns you must follow
- Constraints to be aware of

---

## Initial Documentation

### 7. Update docs/AI_NOTES.md

Add entry for this task:

```markdown
## Current Task
- **Task**: $ARGUMENTS
- **Branch**: <branch-name>
- **Started**: <date>
- **Status**: In Progress

## Key Decisions Made
(To be filled as you work)

## Files Changed
| File | Change Type | Status |
|------|-------------|--------|
| | | |
```

---

## Checklist Summary

Before proceeding to implementation:

- [ ] On main, pulled latest
- [ ] Created feature branch with descriptive name
- [ ] Read relevant CLAUDE.md files
- [ ] Searched for existing patterns
- [ ] Listed key files to understand
- [ ] Checked relevant ADRs
- [ ] Updated AI_NOTES.md with task start

---

## Next Steps

After completing this checklist:

1. **Continue exploration** - Read the key files identified
2. **Create a plan** - Use `/safe-change` for the detailed workflow
3. **Get approval** if needed - For large changes (>3 modules)
4. **Implement** - Following the patterns found

---

## Quick Reference

```bash
# Full start sequence
git checkout main && git pull origin main
git checkout -b feature/<task-slug>

# Then read relevant docs and search for patterns
# Then update docs/AI_NOTES.md
# Then proceed with /safe-change
```
