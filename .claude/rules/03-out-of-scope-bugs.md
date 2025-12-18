# Out-of-Scope Bug Handling

This rule ensures bugs discovered during development are tracked properly.
Never leave bugs untracked. Never expand scope to fix unrelated bugs.

---

## When You Discover a Bug Outside Current Task

If you find a bug that:
- Is NOT in the files you are currently modifying
- Would require scope expansion to fix
- Is tangentially related but not blocking

### DO NOT:

| Action | Why It's Bad |
|--------|--------------|
| Leave it unfixed silently | Creates hidden technical debt |
| Fix it without permission | Scope creep, untested changes |
| Add TODO comment and forget | TODOs get ignored |
| Mention it but take no action | Information lost when context ends |

### DO:

1. **Create a GitHub issue immediately**
   - Use `/create-bug-issue` command
   - Include all relevant details

2. **Document in AI_NOTES.md**
   - Add to "Out-of-Scope Issues Found" table
   - Include issue number after creation

3. **Continue with original task**
   - Don't let the bug derail your work
   - Reference it in PR if related

---

## GitHub Issue Creation

### Required Information

When creating a bug issue, include:

1. **Location**: File path and line number(s)
   ```
   core/risk_manager.py:142-156
   ```

2. **Expected Behavior**: What should happen
   ```
   Position size should respect max_position_size limit
   ```

3. **Actual Behavior**: What actually happens
   ```
   Position size can exceed limit when multiple signals fire
   ```

4. **Reproduction Steps** (if known)
   ```
   1. Set max_position_size to 0.1
   2. Generate rapid signals
   3. Observe position exceeds 0.1
   ```

5. **Discovery Context**: What you were doing
   ```
   Discovered while implementing trailing stop feature
   ```

6. **Potential Fix** (if obvious)
   ```
   Add position check before order execution in OrderManager
   ```

7. **Priority Assessment**
   - **High**: Affects live trading, data integrity, or security
   - **Medium**: Affects functionality but has workaround
   - **Low**: Minor issue, cosmetic, or edge case

### Issue Command

Use the GitHub CLI:

```bash
gh issue create \
  --title "[BUG] <concise description>" \
  --body "<detailed body>" \
  --label "bug,discovered-by-agent,priority:<level>"
```

### Labels

Always apply these labels:

| Label | When to Use |
|-------|-------------|
| `bug` | Always - identifies as bug |
| `discovered-by-agent` | Always - tracks agent-found issues |
| `priority:high` | Affects production/live trading |
| `priority:medium` | Affects functionality, has workaround |
| `priority:low` | Minor issues, edge cases |

---

## Issue Template

```markdown
## Bug Description
<Brief description of the bug>

## Location
- **File**: `<path/to/file.py>`
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
Found while: <what task you were working on>
Related to: <any related issues/PRs>

## Potential Fix
<If obvious, describe approach>

## Priority
- [ ] High - Affects live trading/data integrity
- [ ] Medium - Affects functionality, has workaround
- [ ] Low - Minor issue/edge case
```

---

## Tracking in AI_NOTES.md

After creating the issue, update `docs/AI_NOTES.md`:

```markdown
## Out-of-Scope Issues Found

| Issue # | Description | File | Priority |
|---------|-------------|------|----------|
| #123 | Position size exceeds limit | core/risk_manager.py | High |
| #124 | Missing validation in config | config/settings.py | Low |
```

---

## Decision Tree

```
Found potential bug?
    │
    ├── Is it in files I'm modifying?
    │       │
    │       ├── Yes → Fix it as part of current work
    │       │
    │       └── No → Continue below
    │
    ├── Is it blocking my current work?
    │       │
    │       ├── Yes → Ask user: fix now or work around?
    │       │
    │       └── No → Create issue, continue work
    │
    └── Create GitHub issue with full details
```

---

## Examples

### Example 1: Bug in Unrelated Module

**Scenario**: While implementing trailing stop in `core/risk_manager.py`, you notice a bug in `utils/pnl_calculator.py`.

**Action**:
1. Create issue: `[BUG] PnL calculation incorrect for partial fills`
2. Add to AI_NOTES.md
3. Continue with trailing stop implementation
4. Reference in PR: "Also noticed #123 - unrelated PnL bug"

### Example 2: Bug Blocking Current Work

**Scenario**: Can't implement feature because underlying code is broken.

**Action**:
1. Ask user: "I found a blocking bug in X. Should I fix it first, or should we work around it?"
2. If fix: add to current scope explicitly
3. If workaround: create issue, document workaround, continue

### Example 3: Potential Bug (Uncertain)

**Scenario**: Code looks suspicious but you're not sure it's wrong.

**Action**:
1. Create issue with "potential bug" in title
2. Describe concern and uncertainty
3. Mark as `priority:low` unless obviously critical
