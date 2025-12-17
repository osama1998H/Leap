# ADR 0004: Standardized Checkpoint Format

## Status
Accepted

## Context
Both TransformerPredictor and PPOAgent need to save/load model state. Initially each had its own format, causing:
- Inconsistent checkpoint structure
- Difficult to load models from different training runs
- No backward compatibility handling

## Decision
Create standardized checkpoint format in `utils/checkpoint.py`:

**Standard Keys:**
```python
CHECKPOINT_KEYS = {
    'MODEL_STATE': 'model_state_dict',
    'OPTIMIZER_STATE': 'optimizer_state_dict',
    'CONFIG': 'config',
    'TRAINING_HISTORY': 'training_history',
    'METADATA': 'metadata',
}
```

**Helper Classes:**
- `CheckpointMetadata`: Model type, dimensions, version
- `TrainingHistory`: Unified training metrics

**Functions:**
- `save_checkpoint()`: Save with standardized structure
- `load_checkpoint()`: Load with legacy format support

## Consequences

**Positive:**
- Consistent checkpoint structure
- Backward compatibility with legacy formats
- Training history preserved
- Model metadata for reconstruction

**Negative:**
- Both models must use this format
- Migration needed for old checkpoints (handled automatically)

## Code References
- `utils/checkpoint.py` - Checkpoint utilities
- `models/transformer.py:17-20` - Import pattern
- `models/ppo_agent.py:16-19` - Import pattern
