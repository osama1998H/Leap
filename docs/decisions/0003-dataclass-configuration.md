# ADR 0003: Dataclass-based Configuration

## Status
Accepted

## Context
Configuration needed for:
- Trading environment parameters (EnvConfig)
- System-wide settings (SystemConfig)
- Risk management (RiskConfig)
- Model hyperparameters (TransformerConfig, PPOConfig)

Options considered: YAML files, JSON files, environment variables, dataclasses.

## Decision
Use Python dataclasses with JSON serialization:

**EnvConfig** (`core/trading_types.py`):
- Trading environment configuration
- Factory method: `EnvConfig.from_params()`
- Validation in `__post_init__`

**SystemConfig** (`config/settings.py`):
- Top-level system configuration
- Nested dataclasses for subsystems
- `save()` and `load()` methods for JSON

## Consequences

**Positive:**
- Type safety with IDE support
- Default values in one place
- Validation at construction time
- Easy JSON serialization
- No external dependencies

**Negative:**
- More verbose than YAML/TOML
- Sensitive data must use environment variables

## Code References
- `core/trading_types.py` - EnvConfig
- `config/settings.py` - SystemConfig and nested configs
- Usage: `EnvConfig.from_params(initial_balance=50000.0)`
