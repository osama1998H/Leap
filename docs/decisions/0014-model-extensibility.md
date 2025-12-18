# ADR 0014: Model Extensibility Architecture

## Status
Accepted

## Context

The trading system has tight coupling between components and specific model implementations:

1. **`cli/system.py`** directly imports and instantiates `TransformerPredictor` and `PPOAgent`
2. **`training/trainer.py`** accepts untyped model parameters
3. No abstract interfaces exist for models in `models/`
4. No factory or registry for model creation

This coupling causes problems:

| Issue | Impact |
|-------|--------|
| Direct class imports in 6+ locations | Must modify multiple files to add new model |
| No abstract interface | No contract for what methods a model must implement |
| Hardcoded model types | CLI routing requires modification for each new model |
| No factory pattern | Inconsistent instantiation across codebase |

**Adding a new model (e.g., LSTM predictor) requires:**
- Modifying 9+ files
- ~15 modification points
- 2-3 days effort

## Decision

Introduce model extensibility using Python's structural subtyping (Protocol), following the pattern established in ADR-0010 (Broker Interface Abstraction).

### 1. Model Protocols (`models/base.py`)

```python
@runtime_checkable
class PredictorModel(Protocol):
    """Protocol for prediction models (supervised learning)."""
    input_dim: int
    config: Dict[str, Any]
    device: torch.device

    def train(self, X_train, y_train, X_val, y_val, **kwargs) -> Dict[str, Any]: ...
    def predict(self, X, return_uncertainty=False) -> Dict[str, np.ndarray]: ...
    def online_update(self, X_new, y_new, learning_rate=None) -> float: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

@runtime_checkable
class AgentModel(Protocol):
    """Protocol for RL agents."""
    state_dim: int
    action_dim: int
    config: Dict[str, Any]
    device: torch.device

    def select_action(self, state, deterministic=False) -> Tuple[int, float, float]: ...
    def train_on_env(self, env, total_timesteps, **kwargs) -> Dict[str, Any]: ...
    def online_update(self, n_samples=256, n_epochs=5) -> Optional[Dict[str, float]]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

### 2. Model Factory (`models/factory.py`)

```python
# Global registries
_PREDICTOR_REGISTRY: Dict[str, Type] = {}
_AGENT_REGISTRY: Dict[str, Type] = {}

# Registration decorators
@register_predictor('transformer')
class TransformerPredictor: ...

@register_agent('ppo')
class PPOAgent: ...

# Factory functions
def create_predictor(model_type='transformer', input_dim=None, config=None, device='auto') -> PredictorModel: ...
def create_agent(model_type='ppo', state_dim=None, action_dim=4, config=None, device='auto') -> AgentModel: ...

# Auto-load with type detection
def load_predictor(path: str, device='auto') -> PredictorModel: ...
def load_agent(path: str, device='auto') -> AgentModel: ...

# Discovery
def list_predictors() -> List[str]: ...
def list_agents() -> List[str]: ...
```

### 3. Checkpoint Metadata

Model type stored in checkpoint for auto-detection:
```python
metadata = CheckpointMetadata(
    model_type='transformer',  # or 'lstm', 'gru', etc.
    input_dim=128
)
```

### Why Protocol over ABC?

Using `typing.Protocol` instead of `abc.ABC`:

1. **Structural subtyping**: `TransformerPredictor` and `PPOAgent` satisfy the protocol without modification
2. **Backward compatible**: No changes to existing model implementations required
3. **Better type checking**: mypy understands protocol compliance
4. **Duck typing**: Any class with matching methods works immediately
5. **Consistent with ADR-0010**: Uses same pattern as `BrokerGateway`

## Consequences

### Positive

- **Extensibility**: Add new model by creating class + registering (2 files, 2-3 hours)
- **Type safety**: Protocol type hints enable IDE autocomplete and static analysis
- **Discoverability**: `list_predictors()` and `list_agents()` show available types
- **Auto-loading**: `load_predictor(path)` automatically detects and instantiates correct class
- **Backward compatible**: Direct class usage still works: `TransformerPredictor(input_dim=10)`

### Negative

- **Additional abstraction**: New protocol and factory modules to maintain
- **Runtime duck typing**: Protocol compliance checked at runtime via `isinstance()`
- **Registry management**: Must remember to register new models

## Implementation Files

| File | Purpose |
|------|---------|
| `models/base.py` | Protocol definitions |
| `models/factory.py` | Registry and factory functions |
| `models/__init__.py` | Updated exports |
| `cli/system.py` | Uses factory instead of direct imports |
| `training/trainer.py` | Protocol type hints |
| `tests/test_model_factory.py` | Comprehensive tests |

## Migration Guide

### Adding a New Predictor Model

1. **Create model class** (e.g., `models/lstm.py`):
```python
class LSTMPredictor:
    def __init__(self, input_dim: int, config: dict = None, device: str = 'auto'):
        self.input_dim = input_dim
        self.config = config or {}
        self.device = resolve_device(device)
        # ... initialize LSTM network

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # ... training logic

    def predict(self, X, return_uncertainty=False):
        # ... inference logic

    def online_update(self, X_new, y_new, learning_rate=None):
        # ... online learning

    def save(self, path):
        metadata = CheckpointMetadata(model_type='lstm', input_dim=self.input_dim)
        save_checkpoint(path, self.network.state_dict(), ..., metadata=metadata)

    def load(self, path):
        # ... checkpoint load using utils/checkpoint.py
```

2. **Register model** (in `models/__init__.py` or `models/factory.py`):
```python
from .lstm import LSTMPredictor
register_predictor('lstm')(LSTMPredictor)
```

3. **Use via factory**:
```python
predictor = create_predictor('lstm', input_dim=128, config={...})
```

4. **(Optional) Add config dataclass** (in `config/settings.py`):
```python
@dataclass
class LSTMConfig:
    hidden_size: int = 256
    num_layers: int = 2
```

### Adding a New Agent Model

Same pattern as predictor:

1. Create class implementing `AgentModel` protocol methods
2. Register with `@register_agent('name')` or `register_agent('name')(cls)`
3. Use via `create_agent('name', state_dim=..., action_dim=...)`

## Future Enhancements (Phase 2)

Not in scope for this ADR but enabled by this architecture:

- **CLI model selection**: `--predictor-type lstm --agent-type a2c`
- **Model ensembles**: Combine multiple predictors
- **Additional models**: LSTM, GRU predictors; A2C, SAC agents

## Code References

- `models/base.py` - Protocol definitions
- `models/factory.py` - Registry and factory
- `models/__init__.py` - Module exports
- `cli/system.py:188-227` - Factory usage in `initialize_models()`
- `cli/system.py:1174-1220` - Factory usage in `load_models()`
- `tests/test_model_factory.py` - Test suite
