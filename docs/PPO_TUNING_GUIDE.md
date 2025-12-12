# PPO Agent Tuning Guide for Trading

Based on training diagnostics analysis from MLflow runs.

---

## Diagnostic Interpretation

### Healthy Training Signs

| Metric | Expected Behavior |
|--------|------------------|
| `episode_reward` | Gradually increasing trend |
| `entropy` | Starts high (~1.3), slowly decreases to 0.3-0.6 |
| `value_loss` | Fluctuates, gradually decreases |
| `policy_loss` | Small oscillations around 0 |
| `clip_fraction` | 0.1-0.3 range, not too high or zero |

### Problem Indicators

#### Flat Reward + Near-Zero Losses
**Symptoms:**
- `episode_reward` flat/negative
- `value_loss` → 0 quickly
- `policy_loss` → 0

**Causes:**
1. Reward signal too weak (scaling too low)
2. Advantages near zero
3. Value function predicting constant

**Fix:** Increase reward scaling, check `explained_variance`

#### Entropy Collapse
**Symptoms:**
- `entropy` drops to ~0
- Agent takes same action repeatedly

**Causes:**
1. Unbounded logits growing extreme
2. Learning rate too high
3. Entropy coefficient too low

**Fix:** We've added bounded logits (LOGIT_SCALE=10), should be fixed

#### Training Crashes (NaN)
**Symptoms:**
- Sudden NaN in losses
- Training stops abruptly

**Causes:**
1. Ratio overflow in exp()
2. KL divergence log(0)
3. Extreme reward values

**Fix:** We've added log ratio clamping and KL safety, should be fixed

---

## Recommended Hyperparameters for Trading

### Conservative (Stable, Slower Learning)
```python
config = {
    'learning_rate': 1e-4,      # Lower LR for stability
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.1,        # Tighter clipping
    'entropy_coef': 0.001,      # Low entropy for trading
    'value_coef': 0.5,
    'n_steps': 2048,            # Larger rollouts
    'n_epochs': 10,
    'batch_size': 256,          # Larger batches
    'max_grad_norm': 0.5,
}
```

### Balanced (Recommended Starting Point)
```python
config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.005,      # Reduced from 0.01
    'value_coef': 0.5,
    'n_steps': 1024,
    'n_epochs': 10,
    'batch_size': 128,
    'max_grad_norm': 0.5,
}
```

### Aggressive (Faster Learning, Less Stable)
```python
config = {
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'n_steps': 512,
    'n_epochs': 5,
    'batch_size': 64,
    'max_grad_norm': 1.0,
}
```

---

## Entropy Coefficient for Trading

Trading is different from games - excessive exploration = unnecessary trades and costs.

| entropy_coef | Effect | Use Case |
|--------------|--------|----------|
| 0.01 | High exploration | Early experiments, finding strategies |
| 0.005 | Balanced | Recommended default |
| 0.001 | Low exploration | Fine-tuning, exploitation |
| 0.0 | No entropy bonus | Pure exploitation (risky) |

**Recommendation:** Start with 0.005, decrease to 0.001 once strategy emerges.

---

## Reward Scaling History

| Version | Scaling | Clip | Result |
|---------|---------|------|--------|
| Original | 100x | None | Gradient explosion, training crash |
| Fix v1 | 10x | ±10 | Too weak, agent didn't learn |
| Fix v2 | 50x | ±5 | Balanced signal with stability |

The key insight: **Reward must be large enough to create meaningful advantages, but bounded to prevent numerical instability.**

---

## Additional Diagnostics to Add

For deeper analysis, log these metrics:

```python
# In PPO update, after computing advantages
explained_variance = 1 - (returns - values).var() / returns.var()
# Should be > 0 (ideally 0.5-0.9), negative = value function useless

# Action distribution stats
action_probs = F.softmax(logits, dim=-1)
action_entropy = -(action_probs * action_probs.log()).sum(-1).mean()

# KL divergence (already computed)
approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
# If > 0.02, policy changing too fast → lower LR
```

---

## Troubleshooting Checklist

1. **Reward signal quality**
   - [ ] Rewards vary meaningfully (not constant)
   - [ ] Transaction costs included
   - [ ] Terminal conditions correct

2. **Value function learning**
   - [ ] `explained_variance` > 0
   - [ ] `value_loss` not stuck at 0

3. **Policy updates**
   - [ ] `clip_fraction` in 0.1-0.3 range
   - [ ] `approx_kl` < 0.02

4. **Exploration**
   - [ ] `entropy` not collapsed to 0
   - [ ] Actions distributed, not always same

5. **Numerical stability**
   - [ ] No NaN in any metric
   - [ ] Losses bounded
