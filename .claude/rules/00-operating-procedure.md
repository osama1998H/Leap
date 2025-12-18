# Operating Procedure (MANDATORY)

This rule defines the mandatory workflow for ALL code changes.
Violations of this workflow lead to bugs, scope creep, and architectural drift.

---

## Before ANY Code Changes

### Step 1: Explore (READ-ONLY)

Before writing ANY code, you MUST:

1. **Identify the owning module(s)** for the change
   - Which module(s) will be modified?
   - What are their responsibilities? (check module CLAUDE.md)

2. **Locate existing patterns** already used for the same problem
   - Search for similar implementations in the codebase
   - Do NOT invent new patterns if one already exists

3. **List the 5-10 most relevant files** and WHY each matters
   - Include files that will be modified
   - Include files that demonstrate the pattern to follow
   - Include files that might be affected by the change

4. **Read module-specific CLAUDE.md**
   - `core/CLAUDE.md` for core components
   - `models/CLAUDE.md` for AI models
   - `utils/CLAUDE.md` for utilities
   - `cli/CLAUDE.md` for CLI commands

5. **Check ADRs in `docs/decisions/`**
   - Are there existing decisions that apply?
   - Would this change require a new ADR?

### Step 2: Plan

Before implementing, provide:

1. **Steps with impacted components**
   - List each step and the files it touches
   - Group by module when possible

2. **Architectural risks**
   - Interface changes that affect callers
   - Backward compatibility concerns
   - Side effects (logging, metrics, state)

3. **Test strategy**
   - Which test files to run
   - Exact pytest commands
   - New tests needed

4. **Scope check**
   - If plan affects >3 modules: STOP and get user approval
   - If plan requires new patterns: justify why existing patterns don't work

### Step 3: Implement

When implementing:

1. **Keep diff minimal**
   - Only change what's necessary
   - Don't "clean up" unrelated code

2. **Reuse existing abstractions**
   - Use utilities from `utils/`
   - Use types from `core/trading_types.py`
   - Follow patterns from similar code

3. **NO new patterns if existing pattern exists**
   - If you see the same pattern 2+ times, use it
   - New patterns require justification

4. **Update tests alongside implementation**
   - Don't defer testing
   - Tests prove correctness

### Step 4: Verify

After implementing:

1. **Run targeted tests**
   ```bash
   python -m pytest tests/test_<module>.py -v
   ```

2. **Summarize results**
   - What passed?
   - What failed?
   - What needs follow-up?

3. **If bugs discovered outside current scope**
   - Create GitHub issue immediately: `/create-bug-issue`
   - Do NOT fix them silently (scope creep)
   - Do NOT ignore them (technical debt)

### Step 5: Document

After verification:

1. **Update `docs/AI_NOTES.md`** with implementation state
   - What was done
   - Key decisions made
   - Issues discovered

2. **Update CLAUDE.md** if new patterns introduced
   - Module CLAUDE.md for module-specific patterns
   - Root CLAUDE.md for cross-cutting concerns

3. **Create ADR** if architectural change
   - Use template in `docs/decisions/README.md`
   - Record context, decision, consequences

---

## CRITICAL RULES

### DO NOT start editing until Steps 1 and 2 are completed.

The most common source of agent-induced bugs is editing before understanding.

### DO NOT skip the exploration phase.

Even for "simple" changes, verify assumptions by reading code.

### DO NOT expand scope without permission.

If you discover related issues, create GitHub issues instead of fixing them.

### DO NOT create new abstractions without justification.

Three similar lines of code is better than a premature abstraction.

---

## Quick Reference

| Phase | Goal | Tools |
|-------|------|-------|
| Explore | Understand context | Read, Grep, Glob |
| Plan | Design approach | Text output to user |
| Implement | Make changes | Edit, Write |
| Verify | Confirm correctness | Bash (pytest) |
| Document | Record decisions | Edit (docs) |
