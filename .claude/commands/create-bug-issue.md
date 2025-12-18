# Create GitHub Issue for Discovered Bug

Bug details: $ARGUMENTS

Use this command when you discover a bug outside your current task scope.

---

## Required Information

Gather this information before creating the issue:

### 1. Location
- File path: `<path/to/file.py>`
- Line number(s): `<line or range>`

### 2. Expected Behavior
What SHOULD happen?

### 3. Actual Behavior
What ACTUALLY happens?

### 4. Reproduction Steps
1. Step 1
2. Step 2
3. Observe the bug

### 5. Discovery Context
What were you working on when you found this?

### 6. Priority Assessment
- **High**: Affects live trading, data integrity, or security
- **Medium**: Affects functionality but has workaround
- **Low**: Minor issue, cosmetic, or rare edge case

### 7. Potential Fix (if obvious)
Brief description of how to fix it.

---

## Create the Issue

Run this command:

```bash
gh issue create \
  --title "[BUG] <concise description>" \
  --body "## Bug Description
<Brief description>

## Location
- **File**: \`<path/to/file.py>\`
- **Line(s)**: <line numbers>

## Expected Behavior
<What should happen>

## Actual Behavior
<What actually happens>

## Reproduction Steps
1. <Step 1>
2. <Step 2>
3. <Observe bug>

## Discovery Context
Found while: <what you were working on>

## Potential Fix
<If obvious, describe approach>

## Priority
<High/Medium/Low>
" \
  --label "bug,discovered-by-agent,priority:<level>"
```

Replace:
- `<concise description>` with bug title
- Fill in all sections
- `<level>` with `high`, `medium`, or `low`

---

## After Creating the Issue

- [ ] **Note the issue number** (e.g., #123)

- [ ] **Update `docs/AI_NOTES.md`**:
  Add to "Out-of-Scope Issues Found" table:
  ```markdown
  | #123 | Brief description | file.py | Priority |
  ```

- [ ] **Reference in current PR** if related:
  ```markdown
  Also discovered #123 (unrelated bug in X)
  ```

- [ ] **Continue with original task**
  Don't let this derail your current work.

---

## Example

```bash
gh issue create \
  --title "[BUG] Position size exceeds max limit during rapid signals" \
  --body "## Bug Description
Position size calculation can exceed max_position_size limit when multiple signals fire in rapid succession.

## Location
- **File**: \`core/risk_manager.py\`
- **Line(s)**: 142-156

## Expected Behavior
Position size should never exceed max_position_size (e.g., 0.1 = 10% of balance)

## Actual Behavior
When signals fire rapidly, cumulative position can exceed the limit before the check runs

## Reproduction Steps
1. Set max_position_size to 0.1
2. Generate 5 rapid buy signals in test
3. Observe total position > 10%

## Discovery Context
Found while implementing trailing stop feature in risk_manager.py

## Potential Fix
Add position check before order execution in OrderManager, not just in RiskManager

## Priority
High - Could cause overleveraging in live trading
" \
  --label "bug,discovered-by-agent,priority:high"
```

---

## Quick Reference

| Priority | When to Use | Example |
|----------|-------------|---------|
| High | Live trading, data, security | Position sizing bug |
| Medium | Functionality broken, workaround exists | Wrong metric calculation |
| Low | Minor, cosmetic, edge case | Typo in log message |
