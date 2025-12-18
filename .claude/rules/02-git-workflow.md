# Git Workflow

This rule defines the git workflow for all development work.
Following this workflow ensures clean history and easy code review.

---

## Starting Work

### 1. Always Start from Main

Before starting any new task:

```bash
git checkout main
git pull origin main
```

### 2. Create a Feature Branch

Create a descriptive branch for your work:

```bash
git checkout -b <type>/<description>
```

### Branch Naming Conventions

| Type | Use For | Example |
|------|---------|---------|
| `feature/` | New features | `feature/add-trailing-stop` |
| `fix/` | Bug fixes | `fix/pnl-calculation-error` |
| `refactor/` | Code refactoring | `refactor/broker-abstraction` |
| `docs/` | Documentation only | `docs/update-architecture` |
| `test/` | Test improvements | `test/add-integration-tests` |

### 3. Keep Branch Focused

One branch = One task/feature. If you discover additional work needed:
- Create a GitHub issue for it
- Do NOT expand scope on current branch

---

## During Work

### Commit Frequently

- Make small, logical commits
- Each commit should be a complete thought
- Don't batch unrelated changes

### Commit Message Format

```
<type>: <short description>

<optional body explaining why>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring
- `docs:` - Documentation
- `test:` - Test changes
- `chore:` - Maintenance

**Examples:**
```
feat: add trailing stop loss to risk manager

fix: correct PnL calculation for partial fills

refactor: extract position sizing to utility module

docs: update CLI documentation for adapt command
```

### Keep Working Tree Clean

- Don't leave uncommitted changes overnight
- Use `git stash` if switching context
- Review `git status` before commits

### Pull Updates from Main

For long-running branches:

```bash
git fetch origin
git rebase origin/main
```

---

## Before Finishing

### 1. Ensure All Tests Pass

```bash
python -m pytest tests/ -v
```

### 2. Update Documentation

- Update relevant CLAUDE.md if patterns changed
- Update docstrings for new/changed functions
- Create ADR if architectural decision made

### 3. Review Your Changes

```bash
git diff main...HEAD
```

Ask yourself:
- Does every change relate to the task?
- Are there any unintended changes?
- Is anything missing?

### 4. Create PR with Clear Description

When creating a PR:
- Summarize what changed and why
- List any breaking changes
- Note any follow-up work needed

---

## Branch Protection Rules

### Main Branch

- Never commit directly to main
- All changes via PR
- Require passing tests

### Before Merging

- [ ] All tests pass
- [ ] Documentation updated
- [ ] No scope creep (unrelated changes)
- [ ] PR description complete

---

## Common Scenarios

### Scenario: Discovered Bug While Working

1. Note the bug location and details
2. Run `/create-bug-issue` to create GitHub issue
3. Continue with original task
4. Reference issue in PR if related

### Scenario: Need to Switch Context

1. Commit current work (even WIP)
2. Or stash: `git stash push -m "WIP: description"`
3. Switch branches
4. Return: `git stash pop`

### Scenario: Merge Conflicts

1. `git fetch origin`
2. `git rebase origin/main`
3. Resolve conflicts in each file
4. `git rebase --continue`
5. Force push if needed: `git push --force-with-lease`

### Scenario: Need to Undo Last Commit

```bash
# Keep changes, undo commit
git reset --soft HEAD~1

# Discard changes and commit
git reset --hard HEAD~1
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| Start new work | `git checkout main && git pull && git checkout -b feature/name` |
| Check status | `git status` |
| Stage all | `git add .` |
| Commit | `git commit -m "type: message"` |
| Push | `git push -u origin HEAD` |
| Update from main | `git fetch origin && git rebase origin/main` |
| View changes | `git diff main...HEAD` |
