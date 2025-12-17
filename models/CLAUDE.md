# Models Module Context

The `models/` directory contains the AI/ML models.

## Module Overview

| File | Purpose | Key Classes |
|------|---------|-------------|
| `transformer.py` | Price prediction | `TransformerPredictor`, `TemporalFusionTransformer` |
| `ppo_agent.py` | RL trading agent | `PPOAgent`, `ActorCritic`, `ExperienceBuffer` |

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
