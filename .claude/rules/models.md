---
paths: models/**/*.py
---
# AI Models Development Rules

These rules apply when working on files in the `models/` directory.

---

## Model Architecture Overview

### TransformerPredictor (`models/transformer.py`)
- Temporal Fusion Transformer for price prediction
- Outputs quantile predictions (uncertainty estimation)
- Supports online learning via `partial_fit()` and `adapt()`

### PPOAgent (`models/ppo_agent.py`)
- Proximal Policy Optimization for trading decisions
- Actor-Critic architecture
- Discrete action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
- GAE (Generalized Advantage Estimation) for variance reduction

---

## Required Patterns

### Device Management

Always use `utils/device.py` for device handling:

```python
from utils.device import resolve_device

device = resolve_device(device_str)  # Handles 'auto', 'cuda', 'mps', 'cpu'
```

### Checkpoint Persistence

All model saves/loads MUST use `utils/checkpoint.py`:

```python
from utils.checkpoint import save_checkpoint, load_checkpoint, TrainingHistory, CheckpointMetadata

# Saving
training_history = TrainingHistory(train_losses=losses, val_losses=val_losses)
metadata = CheckpointMetadata(model_type='transformer', input_dim=128)
save_checkpoint(path, model.state_dict(), optimizer.state_dict(), config, training_history, metadata)

# Loading
checkpoint = load_checkpoint(path, device)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Logging

Use standard logging pattern:

```python
import logging
logger = logging.getLogger(__name__)
```

---

## Numerical Stability Considerations

When implementing model components:

1. **Gradient clipping**: Apply gradient clipping to prevent exploding gradients
2. **Loss scaling**: Use appropriate loss scaling for mixed precision
3. **Epsilon values**: Add small epsilon (1e-8) to denominators
4. **Activation bounds**: Be careful with unbounded activations

```python
# Example: Safe division
return value / (denominator + 1e-8)

# Example: Clamped log
return torch.log(torch.clamp(probs, min=1e-8))
```

---

## Online Learning Interface

Both models support online learning:

### TransformerPredictor
```python
# Partial update with new data
model.partial_fit(X_new, y_new, epochs=5)

# Full adaptation
model.adapt(X_new, y_new, learning_rate=1e-5)
```

### PPOAgent
```python
# Update with experience buffer
agent.update(experiences)

# Online training in environment
agent.train_on_env(env, timesteps=1000)
```

---

## Testing Models

When testing model changes:

```bash
# Run model-specific tests
python -m pytest tests/test_integration.py -v -k "model or transformer or ppo"

# Test with small data
python main.py train --symbol EURUSD --epochs 1 --timesteps 1000
```

---

## Common Pitfalls

| Issue | Solution |
|-------|----------|
| NaN losses | Check input normalization, add gradient clipping |
| Memory explosion | Use gradient accumulation, reduce batch size |
| Slow training | Profile with torch.profiler, check data loading |
| Poor convergence | Verify learning rate, check input features |

---

## DO NOT

- Implement trading logic in models (belongs in `core/`)
- Create custom checkpoint formats (use `utils/checkpoint.py`)
- Hardcode device strings (use `resolve_device()`)
- Skip numerical stability checks
