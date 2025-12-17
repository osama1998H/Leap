# ADR 0002: TradingError Exception Hierarchy

## Status
Accepted

## Context
Trading operations can fail in many ways:
- Insufficient funds
- Order rejected by broker
- Position not found
- Connection lost
- Risk limits exceeded

Using generic Python exceptions makes error handling difficult and error messages inconsistent.

## Decision
Create a `TradingError` exception hierarchy in `core/trading_types.py`:

```
TradingError (base)
├── InsufficientFundsError
├── OrderRejectedError
├── PositionError
├── BrokerConnectionError
├── DataPipelineError
└── RiskLimitExceededError
```

**Rules:**
1. All trading-related exceptions inherit from `TradingError`
2. Use `ValueError` for validation/configuration errors
3. Always log exceptions with `logger.exception()`
4. Include context in error messages

## Consequences

**Positive:**
- Unified exception catching: `except TradingError`
- Clear error classification
- Better error messages
- Easier debugging

**Negative:**
- Must remember to use correct exception type
- Some exceptions reserved for future use

## Code References
- `core/trading_types.py:28-111` - Exception hierarchy
- Usage: `from core.trading_types import TradingError, OrderRejectedError`
