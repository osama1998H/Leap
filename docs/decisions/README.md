# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the Leap Trading System.

## What is an ADR?

An ADR documents a significant architectural decision along with its context and consequences. ADRs help AI coding agents and developers understand WHY decisions were made, not just WHAT was implemented.

## When to Create an ADR

Create an ADR when:
- Introducing a new cross-module pattern
- Making a decision that affects multiple modules
- Choosing between significant alternatives
- Establishing a convention that must be followed

## ADR Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| [0001](0001-centralized-utilities.md) | Centralized Utilities Pattern | Accepted | 2024-01 |
| [0002](0002-trading-error-hierarchy.md) | TradingError Exception Hierarchy | Accepted | 2024-01 |
| [0003](0003-dataclass-configuration.md) | Dataclass-based Configuration | Accepted | 2024-01 |
| [0004](0004-standardized-checkpoints.md) | Standardized Checkpoint Format | Accepted | 2024-01 |
| [0005](0005-pnl-position-sizing-utilities.md) | PnL and Position Sizing Utilities | Accepted | 2024-12 |
| [0006](0006-backtest-agent-integration.md) | Backtest-Agent Integration | Accepted | 2024-12 |
| [0007](0007-modular-config-system.md) | Modular Configuration System | Accepted | 2024-12 |
| [0008](0008-cli-package-modularization.md) | CLI Package Modularization | Accepted | 2024-12 |
| [0009](0009-data-persistence.md) | Pipeline Data Persistence | Accepted | 2024-12 |

## ADR Template

```markdown
# ADR-XXXX: [Title]

## Status
[Proposed | Accepted | Deprecated | Superseded by ADR-XXX]

## Context
What is the issue we're facing? What forces are at play?

## Decision
What is the change we're making?

## Consequences
What becomes easier? What becomes harder?

## Code References
- See: `src/path/to/example`
```
