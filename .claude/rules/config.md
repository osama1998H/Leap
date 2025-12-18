---
paths: config/**/*.py
---
# Configuration Module Rules

These rules apply when working on files in the `config/` directory.

---

## Module Overview

The `config/` module contains configuration dataclasses and loaders:

| File | Contents |
|------|----------|
| `settings.py` | All config dataclasses and loaders |
| `__init__.py` | Exports for easy importing |

---

## Configuration Dataclasses

All configurations are Python dataclasses with defaults:

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class MyConfig:
    """Configuration for MyComponent.

    Attributes:
        param1: Description of param1
        param2: Description of param2
    """
    param1: int = 100
    param2: float = 0.01
    param3: List[str] = field(default_factory=list)
    param4: Optional[str] = None
```

### Required Pattern Elements

1. **Docstring**: Describe what the config is for
2. **Type hints**: All fields must have type hints
3. **Defaults**: All fields should have sensible defaults
4. **`field(default_factory=...)`**: For mutable defaults (lists, dicts)

---

## Existing Configuration Classes

| Class | Purpose |
|-------|---------|
| `DataConfig` | Data fetching and symbols |
| `TransformerConfig` | Transformer model architecture |
| `PPOConfig` | PPO agent hyperparameters |
| `RiskConfig` | Risk management parameters |
| `BacktestConfig` | Backtesting settings |
| `AutoTraderConfig` | Live/paper trading settings |
| `LoggingConfig` | Logging configuration |
| `SystemConfig` | Combined system config (legacy) |

---

## Modular Config Loaders

Each config type has a dedicated loader:

```python
from config import (
    load_training_config,     # → (TransformerConfig, PPOConfig, device, seed)
    load_data_config,         # → DataConfig
    load_backtest_config,     # → BacktestConfig
    load_risk_config,         # → RiskConfig
    load_auto_trader_config,  # → AutoTraderConfig
    load_logging_config,      # → LoggingConfig
)

# Usage
data_cfg = load_data_config('config/data.json')
```

### Loader Implementation Pattern

```python
def load_my_config(path: str) -> MyConfig:
    """Load MyConfig from JSON file.

    Args:
        path: Path to JSON configuration file

    Returns:
        MyConfig instance with loaded values

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    with open(path) as f:
        data = json.load(f)
    return MyConfig(**data)
```

---

## JSON Serialization

Configs must support JSON serialization:

```python
@dataclass
class MyConfig:
    param1: int = 100

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'param1': self.param1,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MyConfig':
        """Create from dictionary."""
        return cls(**data)
```

For dataclasses, you can also use:
```python
from dataclasses import asdict
json_data = asdict(config)
```

---

## Adding New Configuration

1. **Define dataclass** in `config/settings.py`:
   ```python
   @dataclass
   class NewFeatureConfig:
       enabled: bool = False
       threshold: float = 0.5
   ```

2. **Add loader function**:
   ```python
   def load_new_feature_config(path: str) -> NewFeatureConfig:
       with open(path) as f:
           data = json.load(f)
       return NewFeatureConfig(**data)
   ```

3. **Export in `__init__.py`**:
   ```python
   from .settings import NewFeatureConfig, load_new_feature_config
   ```

4. **Add example config file** (optional):
   ```json
   // config/new_feature.json
   {
       "enabled": true,
       "threshold": 0.8
   }
   ```

---

## Configuration Resolution

When CLI and config file both specify a value:

```python
# CLI args override config file
final_value = args.my_param if args.my_param is not None else config.my_param
```

Order of precedence (highest to lowest):
1. CLI arguments
2. Config file values
3. Dataclass defaults

---

## Validation

Add validation in `__post_init__`:

```python
@dataclass
class RiskConfig:
    max_position_size: float = 0.1
    max_drawdown: float = 0.2

    def __post_init__(self):
        if not 0 < self.max_position_size <= 1:
            raise ValueError(f"max_position_size must be in (0, 1], got {self.max_position_size}")
        if not 0 < self.max_drawdown <= 1:
            raise ValueError(f"max_drawdown must be in (0, 1], got {self.max_drawdown}")
```

---

## Environment-Specific Overrides

For different environments (dev, prod, test):

```python
# Base config
config = load_data_config('config/data.json')

# Environment override
if os.environ.get('ENV') == 'test':
    config = load_data_config('config/data.test.json')
```

Or merge configs:
```python
base = load_data_config('config/data.json')
override = load_data_config('config/data.local.json')
# Merge override into base
```

---

## DO NOT

- Use mutable defaults without `field(default_factory=...)`
- Skip type hints on config fields
- Create configs without sensible defaults
- Hardcode values that should be configurable
- Skip validation for numerical bounds
