# Safe Change Workflow

Perform a safe change for: $ARGUMENTS

This workflow enforces the mandatory Explore → Plan → Implement → Verify sequence.

---

## Step 1: Explore (NO EDITING)

Before writing ANY code:

- [ ] **Identify owning module(s)**
  - Which module(s) will be modified?
  - What are their responsibilities?

- [ ] **Read module CLAUDE.md**
  - `core/CLAUDE.md` if touching core/
  - `models/CLAUDE.md` if touching models/
  - `utils/CLAUDE.md` if touching utils/
  - `cli/CLAUDE.md` if touching cli/

- [ ] **List 5-10 relevant files with reasons**
  - Files to modify
  - Files with patterns to follow
  - Files that might be affected

- [ ] **Find 2+ existing examples of similar patterns**
  - Search codebase for similar implementations
  - Do NOT invent new patterns if one exists

- [ ] **Check ADRs in `docs/decisions/`**
  - Any existing decisions that apply?
  - Would this require a new ADR?

**OUTPUT**: List the files and patterns found before proceeding.

---

## Step 2: Plan

Before implementing:

- [ ] **List all files to modify**
  - Group by module
  - Note the order of changes

- [ ] **State architectural impacts**
  - Interface changes?
  - Backward compatibility?
  - Side effects?

- [ ] **Define test strategy**
  - Which test files to run?
  - New tests needed?
  - Command: `python -m pytest tests/test_<module>.py -v`

- [ ] **Scope check**
  - If >3 modules affected: STOP and discuss with user
  - If new patterns needed: justify why existing patterns don't work

**OUTPUT**: Present plan to user for approval.

---

## Step 3: Implement

After plan approval:

- [ ] **Make minimal diff**
  - Only change what's necessary
  - Don't refactor unrelated code

- [ ] **Reuse existing patterns**
  - Use utilities from `utils/`
  - Use types from `core/trading_types.py`
  - Follow patterns from exploration

- [ ] **NO new abstractions unless necessary**
  - If you see same pattern 2+ times, use it
  - New patterns require explicit justification

- [ ] **Update tests alongside**
  - Don't defer testing
  - Tests prove correctness

---

## Step 4: Verify

After implementing:

- [ ] **Run tests**
  ```bash
  python -m pytest tests/test_<module>.py -v
  ```

- [ ] **Check for regressions**
  - Any unexpected failures?
  - Any warnings?

- [ ] **Document out-of-scope bugs found**
  - Use `/create-bug-issue` for any bugs discovered
  - Do NOT fix them silently

**OUTPUT**: Test results summary.

---

## Step 5: Document

After verification:

- [ ] **Update `docs/AI_NOTES.md`**
  - What was done
  - Key decisions
  - Issues found

- [ ] **Update relevant CLAUDE.md if new patterns**
  - Module CLAUDE.md for module-specific
  - Root CLAUDE.md for cross-cutting

- [ ] **Create ADR if architectural change**
  - Use template in `docs/decisions/README.md`

---

## CRITICAL REMINDER

**DO NOT start editing until Steps 1 and 2 are complete.**

The most common source of bugs is editing before understanding.
