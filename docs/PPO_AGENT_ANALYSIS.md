# PPO Agent Implementation Analysis Report

**Date:** December 12, 2025
**Files Analyzed:**
- `models/ppo_agent.py` (876 lines)
- `core/trading_env_base.py` (371 lines)
- `core/trading_env.py` (485 lines)
- `tests/test_ppo_agent.py` (671 lines)

---

## Executive Summary

The PPO agent implementation in Leap has several potential bugs and design issues that could significantly impact learning performance. This analysis identifies **12 issues** categorized by severity: 3 Critical, 4 Major, and 5 Moderate.

---

## Critical Issues (3)

### 1. Reward Scaling Creates Numerical Instability

**Location:** `core/trading_env_base.py:229`

```python
return_reward = returns * 100  # Scale to make returns meaningful
```

**Problem:** The 100x scaling on returns, combined with drawdown penalties (50x) and recovery bonuses (25x), creates very large advantage values. When equity changes are significant (e.g., a 5% return), the reward can be `5.0` (500 basis points scaled). This propagates through:
- Large advantages → large policy gradients
- Large gradients → large log probability differences
- Large log prob differences → ratio overflow in PPO update

**Impact:** Training instability, policy divergence, and NaN losses in later training stages.

**Recommendation:** Reduce scaling factors or normalize rewards before computing advantages.

---

### 2. Ratio Overflow/Underflow Risk

**Location:** `models/ppo_agent.py:511` and `models/ppo_agent.py:809`

```python
ratio = torch.exp(new_log_probs - batch_old_log_probs)
```

**Problem:** When the policy changes significantly between collection and update:
- `new_log_probs - batch_old_log_probs` can exceed ±100
- `torch.exp(100)` = `inf`, `torch.exp(-100)` ≈ 0
- This produces `inf` or `nan` in the loss calculation

**Why it happens:**
1. Large rewards cause large policy updates
2. Multiple epochs (n_epochs=10) compound the problem
3. Stale experiences in online learning exacerbate this

**Impact:** Training can suddenly fail with NaN losses, especially during significant market events.

**Recommendation:** Clamp the log ratio before exponentiation:
```python
log_ratio = torch.clamp(new_log_probs - batch_old_log_probs, -20, 20)
ratio = torch.exp(log_ratio)
```

---

### 3. KL Divergence NaN Risk

**Location:** `models/ppo_agent.py:537`

```python
approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
```

**Problem:** When `ratio` approaches 0 (due to extreme policy shifts), `torch.log(ratio)` produces `-inf`, making `approx_kl` = `NaN`.

**Impact:** Corrupted metrics; if KL is used for early stopping (common PPO practice), training logic breaks.

**Recommendation:** Add a minimum clamp:
```python
approx_kl = ((ratio - 1) - torch.log(torch.clamp(ratio, min=1e-8))).mean()
```

---

## Major Issues (4)

### 4. Online Learning Uses Simplified Returns (No GAE)

**Location:** `models/ppo_agent.py:798-801`

```python
# Compute returns (simplified for online learning)
returns = rewards + self.gamma * values * (1 - dones)
advantages = returns - values
```

**Problem:** The online learning method uses one-step returns instead of Generalized Advantage Estimation (GAE). This produces:
- High-variance advantage estimates
- Inconsistent updates compared to main training
- Reduced sample efficiency

**Why it matters:** The main `update()` method uses proper GAE (lines 414-442), but `online_update()` shortcuts this, making online learning ineffective.

**Impact:** Online learning updates are noisy and may not improve the policy meaningfully.

**Recommendation:** Implement GAE for online updates or use a sliding window approach for proper multi-step returns.

---

### 5. Value Bootstrap dtype Mismatch

**Location:** `models/ppo_agent.py:429`

```python
values_extended = torch.cat([values, torch.tensor([last_value], device=self.device)])
```

**Problem:** `last_value` is a Python float (float64 by default). When concatenated with `values` (float32), there's an implicit dtype conversion. On GPU, this can cause:
- Type mismatch warnings/errors
- Subtle precision issues affecting advantage computation

**Impact:** May cause runtime errors on GPU or subtle numerical differences.

**Recommendation:**
```python
values_extended = torch.cat([
    values,
    torch.tensor([last_value], dtype=values.dtype, device=self.device)
])
```

---

### 6. Unbounded Action Logits

**Location:** `models/ppo_agent.py:54-58`

```python
self.actor = nn.Sequential(
    nn.Linear(hidden_sizes[1], hidden_sizes[2]),
    activation(),
    nn.Linear(hidden_sizes[2], action_dim)  # No output bounds
)
```

**Problem:** Actor head outputs can grow unbounded during training. Extreme logits like `[−1000, −1000, 5000, −1000]` make the policy essentially deterministic (softmax → `[0, 0, 1, 0]`), killing exploration.

**How it happens:**
1. Large rewards create large gradients
2. No regularization on output layer
3. Over many updates, logits drift to extremes

**Impact:** Exploration collapse where the agent always takes the same action regardless of state.

**Recommendation:** Consider:
- Tanh activation + scaling on output
- Layer normalization before final linear
- Entropy coefficient scheduling

---

### 7. Inconsistent State Tensor Conversion

**Location:** `models/ppo_agent.py:791-796`

```python
states = torch.FloatTensor([s['state'] for s in samples]).to(self.device)
actions = torch.LongTensor([s['action'] for s in samples]).to(self.device)
```

**Problem:** Creating tensors from lists of numpy arrays via `torch.FloatTensor()` followed by `.to(device)`:
1. Creates tensor on CPU first
2. Then transfers to GPU (inefficient)
3. May have inconsistent dtype handling

Contrast with `select_action()` which uses:
```python
state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
```

**Impact:** Minor inefficiency; potential edge-case dtype issues.

**Recommendation:** Use consistent pattern:
```python
states = torch.as_tensor(np.array([s['state'] for s in samples]), dtype=torch.float32, device=self.device)
```

---

## Moderate Issues (5)

### 8. Advantage Normalization Edge Case

**Location:** `models/ppo_agent.py:472`

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Problem:** If all advantages are identical (e.g., constant reward scenario), `std()` = 0, and division by 1e-8 creates very large values.

**Impact:** Rare but can cause extreme updates in degenerate cases.

**Recommendation:** Check for zero std:
```python
adv_std = advantages.std()
if adv_std > 1e-8:
    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
```

---

### 9. LayerNorm Inconsistency in Network Architecture

**Location:** `models/ppo_agent.py:46-65`

```python
# Feature extractor HAS LayerNorm
self.feature_extractor = self._build_mlp(...)  # Includes LayerNorm

# Actor/Critic heads DO NOT have LayerNorm
self.actor = nn.Sequential(
    nn.Linear(...),
    activation(),
    nn.Linear(...)  # No normalization
)
```

**Problem:** Mixed use of LayerNorm creates inconsistent gradient flow between shared features and separate heads.

**Impact:** May cause training dynamics issues where heads train at different rates than shared features.

---

### 10. Entropy Loss Sign Confusion

**Location:** `models/ppo_agent.py:519-527`

```python
entropy_loss = -entropy.mean()  # Negate entropy

loss = (
    policy_loss +
    self.value_coef * value_loss +
    self.entropy_coef * entropy_loss  # Add negative entropy
)
```

**Problem:** The code computes `entropy_loss = -entropy` then adds it with a positive coefficient. While mathematically correct (we want to maximize entropy, so minimize negative entropy), it's confusing.

**Standard PPO convention:**
```python
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
```

**Impact:** No functional bug, but code clarity issue that could lead to maintenance errors.

---

### 11. Learning Rate Scheduler Timing

**Location:** `models/ppo_agent.py:342-350, 568-570`

```python
# Scheduler T_max calculation
T_max=max(1, self.scheduler_total_steps // self.n_steps)

# Scheduler step (called once per update)
self.scheduler.step()
```

**Problem:** The scheduler steps once per `update()` call, but each update does `n_epochs=10` gradient steps. This means:
- LR decays based on rollout count, not gradient steps
- With `n_epochs=10`, effective LR decay is ~10x slower than intended

**Impact:** LR schedule doesn't match intended training dynamics.

---

### 12. Device Transfer Inefficiency

**Location:** `models/ppo_agent.py:378`

```python
state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
```

**Problem:** Creates tensor on CPU, then transfers to device. For GPU training, this is inefficient (should create directly on device).

**Impact:** Minor performance overhead during action selection.

**Recommendation:**
```python
state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
```

---

## Cascading Failure Analysis

These issues can combine to create training failures:

### Failure Chain 1: Numerical Instability
```
100x reward scaling
    → Large advantages
    → Large policy gradients
    → Extreme log prob differences
    → Ratio overflow (inf/nan)
    → Training crash
```

### Failure Chain 2: Exploration Collapse
```
Large rewards
    → Large gradients to actor
    → Unbounded logits grow extreme
    → Softmax becomes one-hot
    → Policy stuck on single action
    → No learning
```

### Failure Chain 3: Value Function Drift
```
100x rewards
    → Large target values for critic
    → MSE loss explodes
    → Value function diverges
    → Advantages become unreliable
    → Policy updates are random
```

---

## Positive Implementation Aspects

The implementation does include several good practices:

1. **Orthogonal weight initialization** (line 93-94) - Proper for PPO
2. **Gradient clipping** (line 532) - `max_grad_norm=0.5` prevents exploding gradients
3. **Correct GAE implementation** in main training (lines 414-442)
4. **Pre-allocated rollout buffers** (lines 169-201) - Performance optimization
5. **Early stopping with patience** (lines 706-735) - Prevents overfitting
6. **Empty buffer guard** (lines 452-461) - Handles edge case

---

## Priority Recommendations

### Immediate (Critical fixes)
1. **Clamp log ratios** before exponentiation to prevent overflow
2. **Fix KL calculation** to handle zero/near-zero ratios
3. **Review reward scaling** - consider reducing or normalizing

### Short-term (Major fixes)
4. **Fix dtype mismatch** in value bootstrap
5. **Add logit regularization** or output bounds to actor
6. **Implement proper GAE** for online learning

### Medium-term (Architecture improvements)
7. Consistent LayerNorm usage across network
8. Fix scheduler timing to match gradient steps
9. Standardize tensor creation patterns

---

## Testing Gaps

The test suite (`tests/test_ppo_agent.py`) has good coverage but is missing:

1. **NaN handling tests** - No tests for numerical overflow scenarios
2. **Extreme reward tests** - The `test_extreme_rewards` uses ±1000, but real scaling could produce larger values
3. **GPU-specific tests** - dtype mismatch may only manifest on GPU
4. **Long training stability** - Tests use short runs (200 timesteps)

**Recommended additional tests:**
```python
def test_ratio_overflow_handling():
    """Test that extreme policy shifts don't cause NaN."""
    # Store experiences with very stale log_probs
    # Verify update completes without NaN

def test_advantage_zero_std():
    """Test handling when all advantages are identical."""
    # All same reward scenario

def test_long_training_stability():
    """Test training stability over 10k+ timesteps."""
    # Verify no NaN accumulation
```

---

## Conclusion

The PPO implementation has a solid foundation with proper GAE, gradient clipping, and good buffer management. However, the **reward scaling combined with lack of ratio clamping** creates a significant risk of numerical instability during training. The **online learning simplification** also limits the system's ability to adapt in production.

Addressing the three critical issues should be the immediate priority, as they can cause training to fail completely during significant market movements.
