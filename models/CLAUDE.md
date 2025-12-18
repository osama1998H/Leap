# Models Module Context

The `models/` directory contains the AI/ML models.

## Module Overview

| File | Purpose | Key Classes |
|------|---------|-------------|
| `base.py` | Model protocols | `PredictorModel`, `AgentModel` |
| `factory.py` | Model factory | `create_predictor()`, `create_agent()`, `register_predictor()`, `register_agent()` |
| `transformer.py` | Price prediction | `TransformerPredictor`, `TemporalFusionTransformer` |
| `ppo_agent.py` | RL trading agent | `PPOAgent`, `ActorCritic`, `ExperienceBuffer` |

## Model Extensibility (ADR-0014)

The models module uses Protocol-based extensibility for easy addition of new model types.

### Creating Models via Factory

```python
from models import create_predictor, create_agent

# Create predictor
predictor = create_predictor('transformer', input_dim=128, config={...})

# Create agent
agent = create_agent('ppo', state_dim=72, action_dim=4, config={...})

# Auto-load from checkpoint
predictor = load_predictor('saved_models/predictor.pt')
agent = load_agent('saved_models/agent.pt')

# List available types
print(list_predictors())  # ['transformer']
print(list_agents())      # ['ppo']
```

### Adding a New Model

1. Create model class implementing protocol methods
2. Register with decorator or explicit call
3. Use via factory

```python
from models import register_predictor

@register_predictor('lstm')
class LSTMPredictor:
    def __init__(self, input_dim: int, config: dict = None, device: str = 'auto'):
        ...

    def train(self, X_train, y_train, ...): ...
    def predict(self, X, return_uncertainty=False): ...
    def online_update(self, X_new, y_new, learning_rate=None): ...
    def save(self, path: str): ...
    def load(self, path: str): ...
```

### Protocol Type Hints

For type-safe code that accepts any model:

```python
from models import PredictorModel, AgentModel

def train_pipeline(predictor: PredictorModel, agent: AgentModel):
    predictor.train(...)
    agent.train_on_env(...)
```

## Architecture

### TransformerPredictor
- Temporal Fusion Transformer architecture
- Quantile predictions (0.1, 0.5, 0.9) for uncertainty
- Online learning support (learning rate: 1e-5)
- Main learning rate: 1e-4

### PPOAgent
- Proximal Policy Optimization with Actor-Critic
- Action space: HOLD(0), BUY(1), SELL(2), CLOSE(3)
- GAE for advantage estimation (gamma=0.99, lambda=0.95)
- Clip epsilon: 0.2, entropy coefficient: 0.01

## Required Patterns

### Device Management
```python
from utils.device import resolve_device
device = resolve_device('auto')  # Returns cuda/mps/cpu
```

### Checkpoints
```python
from utils.checkpoint import save_checkpoint, load_checkpoint

# Save
save_checkpoint(path, model.state_dict(), optimizer.state_dict(),
                config, training_history, metadata)

# Load
checkpoint = load_checkpoint(path, device)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)
```

## Common Gotchas

1. **Input dimensions**: Must match between training and inference
2. **Window size**: Default 60, must be consistent with EnvConfig
3. **Checkpoint compatibility**: Use standardized format for new saves
4. **Numerical stability**: Constants defined at module level (LOG_RATIO_CLAMP, etc.)
