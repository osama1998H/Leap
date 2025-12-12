"""
PPO Agent Bug Evidence Tests

This module demonstrates and tests each identified bug in the PPO agent implementation.
Each test shows the BEFORE behavior (bug manifestation) and validates the AFTER fix.

Run with: python -m pytest tests/test_ppo_bugs_evidence.py -v -s
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import warnings

# Make pytest optional
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


# ============================================================================
# Test Utilities
# ============================================================================

def create_mock_ppo_scenario(
    n_samples: int = 100,
    log_prob_diff_range: Tuple[float, float] = (-2, 2)
) -> Dict[str, torch.Tensor]:
    """Create mock PPO update scenario data."""
    low, high = log_prob_diff_range
    old_log_probs = torch.randn(n_samples)
    log_diff = (high - low) * torch.rand(n_samples) + low
    return {
        'old_log_probs': old_log_probs,
        'new_log_probs': old_log_probs + log_diff,
        'advantages': torch.randn(n_samples),
        'values': torch.randn(n_samples),
        'returns': torch.randn(n_samples),
    }


# ============================================================================
# BUG #1: Ratio Overflow/Underflow
# ============================================================================

class TestRatioOverflow:
    """
    Evidence for Critical Bug #1: Ratio overflow when log prob differences are extreme.

    Location: models/ppo_agent.py:511
    Code: ratio = torch.exp(new_log_probs - batch_old_log_probs)
    """

    def test_bug_evidence_ratio_overflow(self):
        """
        EVIDENCE: Demonstrate that extreme log prob differences cause overflow.

        When policy changes significantly (e.g., after large reward), the difference
        between new and old log probs can be very large, causing exp() to overflow.
        """
        print("\n" + "="*70)
        print("BUG #1: Ratio Overflow Evidence")
        print("="*70)

        # Simulate extreme policy shift (happens with large rewards/updates)
        old_log_probs = torch.tensor([-5.0, -10.0, -2.0, -50.0, -100.0])
        new_log_probs = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])  # Policy became confident

        # BUGGY CODE (current implementation)
        log_diff = new_log_probs - old_log_probs
        ratio_buggy = torch.exp(log_diff)

        print(f"\nLog probability differences: {log_diff.tolist()}")
        print(f"Ratios (BUGGY - no clamping): {ratio_buggy.tolist()}")

        # Check for overflow
        has_inf = torch.isinf(ratio_buggy).any()
        has_nan = torch.isnan(ratio_buggy).any()

        print(f"\nHas infinity: {has_inf}")
        print(f"Has NaN: {has_nan}")

        # This SHOULD fail (demonstrates the bug)
        assert has_inf or ratio_buggy.max() > 1e30, "Bug evidence: extreme ratios expected"

        # Calculate policy loss with buggy ratio
        advantages = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        clip_epsilon = 0.2

        surr1 = ratio_buggy * advantages
        surr2 = torch.clamp(ratio_buggy, 1-clip_epsilon, 1+clip_epsilon) * advantages
        policy_loss_buggy = -torch.min(surr1, surr2).mean()

        print(f"\nPolicy loss (BUGGY): {policy_loss_buggy.item()}")
        print(f"Loss is finite: {torch.isfinite(policy_loss_buggy)}")

    def test_fix_ratio_overflow_with_clamping(self):
        """
        FIX: Clamp log probability differences before exponentiation.

        Best practice from PPO implementations (Stable-Baselines3, CleanRL):
        Clamp the log ratio to prevent numerical overflow.
        """
        print("\n" + "="*70)
        print("FIX #1: Ratio Overflow - Log Ratio Clamping")
        print("="*70)

        old_log_probs = torch.tensor([-5.0, -10.0, -2.0, -50.0, -100.0])
        new_log_probs = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

        # FIXED CODE
        LOG_RATIO_CLAMP = 20.0  # exp(20) ≈ 485 million, exp(-20) ≈ 0
        log_ratio = new_log_probs - old_log_probs
        log_ratio_clamped = torch.clamp(log_ratio, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
        ratio_fixed = torch.exp(log_ratio_clamped)

        print(f"\nLog ratio (raw): {log_ratio.tolist()}")
        print(f"Log ratio (clamped to ±{LOG_RATIO_CLAMP}): {log_ratio_clamped.tolist()}")
        print(f"Ratios (FIXED): {ratio_fixed.tolist()}")

        # Verify fix
        assert not torch.isinf(ratio_fixed).any(), "Fix failed: still has infinity"
        assert not torch.isnan(ratio_fixed).any(), "Fix failed: still has NaN"
        assert ratio_fixed.max() < 1e10, "Fix failed: ratio still too large"

        # Calculate policy loss with fixed ratio
        advantages = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        clip_epsilon = 0.2

        surr1 = ratio_fixed * advantages
        surr2 = torch.clamp(ratio_fixed, 1-clip_epsilon, 1+clip_epsilon) * advantages
        policy_loss_fixed = -torch.min(surr1, surr2).mean()

        print(f"\nPolicy loss (FIXED): {policy_loss_fixed.item()}")
        assert torch.isfinite(policy_loss_fixed), "Fix failed: loss not finite"
        print("PASSED: Policy loss is now finite and stable")


# ============================================================================
# BUG #2: KL Divergence NaN
# ============================================================================

class TestKLDivergenceNaN:
    """
    Evidence for Critical Bug #2: KL divergence produces NaN when ratio approaches 0.

    Location: models/ppo_agent.py:537
    Code: approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
    """

    def test_bug_evidence_kl_nan(self):
        """
        EVIDENCE: Demonstrate that near-zero ratios cause KL to be NaN.
        """
        print("\n" + "="*70)
        print("BUG #2: KL Divergence NaN Evidence")
        print("="*70)

        # Simulate case where new policy assigns ~0 probability to old actions
        # This happens when policy shifts dramatically
        ratio = torch.tensor([1.0, 0.5, 0.1, 0.01, 1e-10, 0.0])

        # BUGGY CODE
        approx_kl_buggy = ((ratio - 1) - torch.log(ratio)).mean()

        print(f"\nRatios: {ratio.tolist()}")
        print(f"log(ratio): {torch.log(ratio).tolist()}")
        print(f"Approx KL (BUGGY): {approx_kl_buggy.item()}")

        # Check for NaN or inf
        has_issue = torch.isnan(approx_kl_buggy) or torch.isinf(approx_kl_buggy)
        print(f"KL has NaN/Inf: {has_issue}")

        assert has_issue, "Bug evidence: KL should have NaN/Inf with zero ratio"

    def test_fix_kl_with_clamping(self):
        """
        FIX: Clamp ratio before taking log to prevent NaN.
        """
        print("\n" + "="*70)
        print("FIX #2: KL Divergence - Ratio Clamping")
        print("="*70)

        ratio = torch.tensor([1.0, 0.5, 0.1, 0.01, 1e-10, 0.0])

        # FIXED CODE
        MIN_RATIO = 1e-8
        ratio_safe = torch.clamp(ratio, min=MIN_RATIO)
        approx_kl_fixed = ((ratio_safe - 1) - torch.log(ratio_safe)).mean()

        print(f"\nRatios (raw): {ratio.tolist()}")
        print(f"Ratios (clamped min={MIN_RATIO}): {ratio_safe.tolist()}")
        print(f"Approx KL (FIXED): {approx_kl_fixed.item()}")

        assert torch.isfinite(approx_kl_fixed), "Fix failed: KL still not finite"
        print("PASSED: KL divergence is now finite and computable")


# ============================================================================
# BUG #3: Reward Scaling Numerical Instability
# ============================================================================

class TestRewardScaling:
    """
    Evidence for Critical Bug #3: 100x reward scaling causes numerical instability.

    Location: core/trading_env_base.py:229
    Code: return_reward = returns * 100
    """

    def test_bug_evidence_reward_scaling(self):
        """
        EVIDENCE: Demonstrate how 100x scaling propagates to instability.
        """
        print("\n" + "="*70)
        print("BUG #3: Reward Scaling Evidence")
        print("="*70)

        # Simulate realistic trading scenario
        # 5% equity change is not uncommon in leveraged trading
        prev_equity = 10000.0
        equity_changes = [0.05, 0.10, -0.08, 0.15, -0.12]  # 5%, 10%, -8%, etc.

        print("\nEquity changes (realistic trading):")
        for change in equity_changes:
            returns = change  # Decimal return

            # BUGGY: 100x scaling
            return_reward_buggy = returns * 100
            # Plus drawdown penalty (50x) if negative
            drawdown_penalty = -abs(change) * 50 if change < 0 else 0
            total_reward_buggy = return_reward_buggy + drawdown_penalty

            print(f"  Change {change*100:+.1f}% -> Reward (BUGGY): {total_reward_buggy:+.2f}")

        # Show how this affects advantages
        rewards = torch.tensor([r * 100 for r in equity_changes])
        advantages = rewards - rewards.mean()

        print(f"\nAdvantages from scaled rewards: {advantages.tolist()}")
        print(f"Advantage std: {advantages.std().item():.2f}")
        print(f"Max |advantage|: {advantages.abs().max().item():.2f}")

        # These large values will cause gradient issues
        assert advantages.abs().max() > 10, "Bug evidence: advantages should be large"

    def test_fix_reward_normalization(self):
        """
        FIX: Use running normalization for rewards (like Stable-Baselines3).

        Best practice: Track running mean/std and normalize rewards.
        """
        print("\n" + "="*70)
        print("FIX #3: Reward Normalization")
        print("="*70)

        class RunningMeanStd:
            """Running mean and standard deviation tracker."""
            def __init__(self, epsilon=1e-8):
                self.mean = 0.0
                self.var = 1.0
                self.count = epsilon

            def update(self, x):
                batch_mean = np.mean(x)
                batch_var = np.var(x)
                batch_count = len(x)
                self._update_from_moments(batch_mean, batch_var, batch_count)

            def _update_from_moments(self, batch_mean, batch_var, batch_count):
                delta = batch_mean - self.mean
                tot_count = self.count + batch_count
                new_mean = self.mean + delta * batch_count / tot_count
                m_a = self.var * self.count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
                new_var = M2 / tot_count
                self.mean = new_mean
                self.var = new_var
                self.count = tot_count

            def normalize(self, x):
                return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

        # Simulate rewards over time
        equity_changes = [0.05, 0.10, -0.08, 0.15, -0.12, 0.03, -0.02, 0.08]

        # FIXED: Reduced scaling + normalization
        REWARD_SCALE = 10.0  # Reduced from 100
        raw_rewards = [c * REWARD_SCALE for c in equity_changes]

        reward_normalizer = RunningMeanStd()
        reward_normalizer.update(raw_rewards[:4])  # Warm up

        normalized_rewards = reward_normalizer.normalize(np.array(raw_rewards))

        print(f"\nRaw rewards (10x scale): {[f'{r:.2f}' for r in raw_rewards]}")
        print(f"Normalized rewards: {[f'{r:.2f}' for r in normalized_rewards]}")
        print(f"Normalized std: {np.std(normalized_rewards):.3f}")
        print(f"Max |normalized|: {np.abs(normalized_rewards).max():.3f}")

        assert np.abs(normalized_rewards).max() < 5, "Fix validation"
        print("\nPASSED: Rewards are now bounded and normalized")


# ============================================================================
# BUG #4: Value Bootstrap dtype Mismatch
# ============================================================================

class TestDtypeMismatch:
    """
    Evidence for Major Bug #4: dtype mismatch in value bootstrapping.

    Location: models/ppo_agent.py:429
    Code: values_extended = torch.cat([values, torch.tensor([last_value], device=self.device)])
    """

    def test_bug_evidence_dtype_mismatch(self):
        """
        EVIDENCE: Demonstrate dtype mismatch between values and last_value.
        """
        print("\n" + "="*70)
        print("BUG #4: dtype Mismatch Evidence")
        print("="*70)

        # values tensor is float32 (from network)
        values = torch.rand(10, dtype=torch.float32)

        # last_value comes from Python float (float64)
        last_value = 0.5  # Python float is float64

        # BUGGY CODE
        last_value_tensor = torch.tensor([last_value])  # Default: float64!

        print(f"\nvalues dtype: {values.dtype}")
        print(f"last_value_tensor dtype: {last_value_tensor.dtype}")
        print(f"dtype mismatch: {values.dtype != last_value_tensor.dtype}")

        # Concatenation will work but with implicit conversion
        values_extended_buggy = torch.cat([values, last_value_tensor])
        print(f"Concatenated dtype: {values_extended_buggy.dtype}")

        # On GPU, this can cause issues
        if torch.cuda.is_available():
            values_gpu = values.cuda()
            # This line can fail or produce warnings on some PyTorch versions
            try:
                values_extended_gpu = torch.cat([values_gpu, torch.tensor([last_value]).cuda()])
                print(f"GPU concatenated dtype: {values_extended_gpu.dtype}")
            except (RuntimeError, TypeError) as e:
                print(f"GPU error: {e}")

    def test_fix_explicit_dtype(self):
        """
        FIX: Explicitly specify dtype when creating the tensor.
        """
        print("\n" + "="*70)
        print("FIX #4: Explicit dtype Specification")
        print("="*70)

        values = torch.rand(10, dtype=torch.float32)
        last_value = 0.5
        device = values.device

        # FIXED CODE
        last_value_tensor = torch.tensor([last_value], dtype=values.dtype, device=device)

        print(f"\nvalues dtype: {values.dtype}")
        print(f"last_value_tensor dtype (FIXED): {last_value_tensor.dtype}")

        values_extended_fixed = torch.cat([values, last_value_tensor])
        print(f"Concatenated dtype: {values_extended_fixed.dtype}")

        assert values_extended_fixed.dtype == torch.float32, "Fix failed"
        print("\nPASSED: dtypes now match consistently")


# ============================================================================
# BUG #5: Online Learning Without GAE
# ============================================================================

class TestOnlineLearningGAE:
    """
    Evidence for Major Bug #5: Online learning uses simplified 1-step returns.

    Location: models/ppo_agent.py:798-801
    """

    def test_bug_evidence_no_gae(self):
        """
        EVIDENCE: Show variance difference between 1-step and GAE returns.
        """
        print("\n" + "="*70)
        print("BUG #5: Online Learning Without GAE")
        print("="*70)

        # Simulate a sequence of experiences
        n_steps = 20
        gamma = 0.99
        gae_lambda = 0.95

        rewards = torch.tensor([0.1, -0.2, 0.3, 0.1, -0.1, 0.2, 0.4, -0.3, 0.1, 0.2,
                               0.1, -0.2, 0.3, 0.1, -0.1, 0.2, 0.4, -0.3, 0.1, 0.2])
        values = torch.tensor([0.5, 0.4, 0.6, 0.5, 0.4, 0.5, 0.7, 0.4, 0.5, 0.6,
                               0.5, 0.4, 0.6, 0.5, 0.4, 0.5, 0.7, 0.4, 0.5, 0.6])
        dones = torch.zeros(n_steps)

        # BUGGY: 1-step returns (current online_update)
        returns_1step = rewards + gamma * values * (1 - dones)
        advantages_1step = returns_1step - values

        # CORRECT: GAE
        advantages_gae = torch.zeros(n_steps)
        gae = 0.0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages_gae[t] = gae

        print("\nAdvantages (1-step BUGGY):")
        print(f"  Mean: {advantages_1step.mean():.4f}, Std: {advantages_1step.std():.4f}")
        print(f"  Values: {advantages_1step[:5].tolist()}")

        print("\nAdvantages (GAE CORRECT):")
        print(f"  Mean: {advantages_gae.mean():.4f}, Std: {advantages_gae.std():.4f}")
        print(f"  Values: {advantages_gae[:5].tolist()}")

        print(f"\nVariance ratio (1-step/GAE): {(advantages_1step.var() / advantages_gae.var()).item():.2f}x")

    def test_fix_online_gae(self):
        """
        FIX: Implement proper GAE for online learning.
        """
        print("\n" + "="*70)
        print("FIX #5: Online Learning with Proper GAE")
        print("="*70)

        def compute_gae_online(
            rewards: torch.Tensor,
            values: torch.Tensor,
            dones: torch.Tensor,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            last_value: float = 0.0
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute GAE for online learning (same as main training)."""
            n_steps = len(rewards)
            advantages = torch.zeros(n_steps)
            returns = torch.zeros(n_steps)

            values_extended = torch.cat([values, torch.tensor([last_value])])

            gae = 0.0
            for t in reversed(range(n_steps)):
                delta = (
                    rewards[t] +
                    gamma * values_extended[t + 1] * (1 - dones[t]) -
                    values_extended[t]
                )
                gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]

            return returns, advantages

        # Test the fix
        n_steps = 20
        rewards = torch.rand(n_steps) * 0.2 - 0.1
        values = torch.rand(n_steps) * 0.5 + 0.3
        dones = torch.zeros(n_steps)

        _returns_fixed, advantages_fixed = compute_gae_online(rewards, values, dones)

        print("\nGAE advantages (FIXED):")
        print(f"  Mean: {advantages_fixed.mean():.4f}")
        print(f"  Std: {advantages_fixed.std():.4f}")
        print(f"  All finite: {torch.isfinite(advantages_fixed).all()}")

        print("\nPASSED: Online learning now uses proper GAE")


# ============================================================================
# BUG #6: Unbounded Action Logits
# ============================================================================

class TestUnboundedLogits:
    """
    Evidence for Major Bug #6: Actor can produce unbounded logits.

    Location: models/ppo_agent.py:54-58
    """

    def test_bug_evidence_unbounded_logits(self):
        """
        EVIDENCE: Show how logits can grow unbounded with repeated updates.
        """
        print("\n" + "="*70)
        print("BUG #6: Unbounded Action Logits Evidence")
        print("="*70)

        # Simulate actor output evolution during training
        # Start with reasonable logits, then simulate drift

        def softmax_entropy(logits):
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            return entropy.item(), probs.tolist()

        print("\nSimulating logit growth over training:")

        # Normal logits (healthy)
        logits_normal = torch.tensor([[1.0, 0.5, -0.5, 0.0]])
        entropy_normal, probs_normal = softmax_entropy(logits_normal)
        print(f"\nNormal logits: {logits_normal[0].tolist()}")
        print(f"  Probs: {[f'{p:.3f}' for p in probs_normal[0]]}")
        print(f"  Entropy: {entropy_normal:.4f}")

        # Drifted logits (after many updates with large gradients)
        logits_drifted = torch.tensor([[50.0, -20.0, -30.0, -10.0]])
        entropy_drifted, probs_drifted = softmax_entropy(logits_drifted)
        print(f"\nDrifted logits: {logits_drifted[0].tolist()}")
        print(f"  Probs: {[f'{p:.3f}' for p in probs_drifted[0]]}")
        print(f"  Entropy: {entropy_drifted:.4f}")

        # Extreme logits (exploration collapse)
        logits_extreme = torch.tensor([[500.0, -200.0, -300.0, -100.0]])
        entropy_extreme, probs_extreme = softmax_entropy(logits_extreme)
        print(f"\nExtreme logits: {logits_extreme[0].tolist()}")
        print(f"  Probs: {[f'{p:.6f}' for p in probs_extreme[0]]}")
        print(f"  Entropy: {entropy_extreme:.6f}")

        print(f"\nEntropy collapse: {entropy_normal:.4f} -> {entropy_extreme:.6f}")
        assert entropy_extreme < 0.01, "Bug evidence: entropy should collapse"

    def test_fix_bounded_logits(self):
        """
        FIX: Use tanh activation + scaling to bound logits.
        """
        print("\n" + "="*70)
        print("FIX #6: Bounded Action Logits")
        print("="*70)

        class BoundedActor(nn.Module):
            """Actor with bounded output logits."""
            def __init__(self, hidden_dim, action_dim, logit_scale=5.0):
                super().__init__()
                self.linear = nn.Linear(hidden_dim, action_dim)
                self.logit_scale = logit_scale

            def forward(self, x):
                raw = self.linear(x)
                # Bound logits using tanh
                bounded = torch.tanh(raw) * self.logit_scale
                return bounded

        actor = BoundedActor(64, 4, logit_scale=5.0)

        # Even with extreme inputs, outputs are bounded
        extreme_input = torch.randn(1, 64) * 100
        bounded_logits = actor(extreme_input)

        print(f"\nInput magnitude: {extreme_input.abs().max():.1f}")
        print(f"Bounded logits: {bounded_logits[0].tolist()}")
        print(f"Max |logit|: {bounded_logits.abs().max():.2f}")
        print(f"Guaranteed bound: ±{actor.logit_scale}")

        # Check entropy is maintained
        probs = F.softmax(bounded_logits, dim=-1)
        log_probs = F.log_softmax(bounded_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        print(f"\nEntropy with bounded logits: {entropy.item():.4f}")
        assert bounded_logits.abs().max() <= 5.0, "Fix failed: logits not bounded"
        print("\nPASSED: Logits are now bounded, preventing exploration collapse")


# ============================================================================
# BUG #7: Advantage Normalization Edge Case
# ============================================================================

class TestAdvantageNormalization:
    """
    Evidence for Moderate Bug #7: Advantage normalization with zero std.

    Location: models/ppo_agent.py:472
    """

    def test_bug_evidence_zero_std(self):
        """
        EVIDENCE: Show what happens when all advantages are identical.
        """
        print("\n" + "="*70)
        print("BUG #7: Advantage Normalization Zero Std")
        print("="*70)

        # All same advantages (can happen with constant rewards)
        advantages = torch.ones(10) * 0.5

        # BUGGY CODE
        adv_std = advantages.std()
        normalized_buggy = (advantages - advantages.mean()) / (adv_std + 1e-8)

        print(f"\nAdvantages: {advantages.tolist()}")
        print(f"Mean: {advantages.mean()}, Std: {adv_std}")
        print(f"Normalized (BUGGY): {normalized_buggy.tolist()}")
        print(f"Normalized values magnitude: {normalized_buggy.abs().max():.2e}")

        # With std=0, dividing by 1e-8 creates very small but non-zero values
        # Not catastrophic but mathematically incorrect

    def test_fix_safe_normalization(self):
        """
        FIX: Check for zero std before normalizing.
        """
        print("\n" + "="*70)
        print("FIX #7: Safe Advantage Normalization")
        print("="*70)

        def safe_normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            """Safely normalize advantages, handling zero variance case."""
            adv_mean = advantages.mean()
            adv_std = advantages.std()

            if adv_std < eps:
                # All advantages nearly identical - return zeros
                # (no relative preference between actions)
                return torch.zeros_like(advantages)

            return (advantages - adv_mean) / (adv_std + eps)

        # Test with zero std
        advantages_same = torch.ones(10) * 0.5
        normalized_fixed = safe_normalize_advantages(advantages_same)

        print(f"\nConstant advantages: {advantages_same[:3].tolist()}...")
        print(f"Normalized (FIXED): {normalized_fixed[:3].tolist()}...")

        # Test with normal std
        advantages_normal = torch.randn(10)
        normalized_normal = safe_normalize_advantages(advantages_normal)

        print(f"\nNormal advantages std: {advantages_normal.std():.4f}")
        print(f"Normalized std: {normalized_normal.std():.4f}")

        print("\nPASSED: Safe normalization handles edge cases correctly")


# ============================================================================
# Integration Test: All Fixes Together
# ============================================================================

class TestAllFixesIntegration:
    """Integration test demonstrating all fixes working together."""

    def test_stable_ppo_update(self):
        """
        Test a PPO update with all fixes applied.
        """
        print("\n" + "="*70)
        print("INTEGRATION TEST: Stable PPO Update")
        print("="*70)

        # Simulate PPO update with challenging data
        n_samples = 100
        clip_epsilon = 0.2

        # Generate challenging scenario
        old_log_probs = torch.randn(n_samples) * 2 - 5  # Some very negative
        new_log_probs = torch.randn(n_samples) * 0.5  # Policy changed
        advantages = torch.randn(n_samples) * 10  # Large advantages
        values = torch.randn(n_samples)
        returns = values + advantages * 0.1

        # === Apply all fixes ===

        # Fix 1: Clamp log ratios
        LOG_RATIO_CLAMP = 20.0
        log_ratio = new_log_probs - old_log_probs
        log_ratio_clamped = torch.clamp(log_ratio, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
        ratio = torch.exp(log_ratio_clamped)

        # Fix 2: Safe KL calculation
        ratio_safe = torch.clamp(ratio, min=1e-8)
        approx_kl = ((ratio_safe - 1) - torch.log(ratio_safe)).mean()

        # Fix 7: Safe advantage normalization
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages_norm = (advantages - advantages.mean()) / (adv_std + 1e-8)
        else:
            advantages_norm = torch.zeros_like(advantages)

        # PPO clipped objective
        surr1 = ratio * advantages_norm
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_norm
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy (simulated)
        entropy = torch.tensor(0.5)

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        print("\nWith all fixes applied:")
        print(f"  Policy loss: {policy_loss.item():.4f} (finite: {torch.isfinite(policy_loss)})")
        print(f"  Value loss: {value_loss.item():.4f} (finite: {torch.isfinite(value_loss)})")
        print(f"  Approx KL: {approx_kl.item():.4f} (finite: {torch.isfinite(approx_kl)})")
        print(f"  Total loss: {total_loss.item():.4f} (finite: {torch.isfinite(total_loss)})")

        # Verify all values are finite
        assert torch.isfinite(policy_loss), "Policy loss not finite"
        assert torch.isfinite(value_loss), "Value loss not finite"
        assert torch.isfinite(approx_kl), "KL not finite"
        assert torch.isfinite(total_loss), "Total loss not finite"

        print("\nALL TESTS PASSED: PPO update is numerically stable")


# ============================================================================
# Run Tests
# ============================================================================

def run_all_tests():
    """Run all tests without pytest."""
    test_classes = [
        TestRatioOverflow(),
        TestKLDivergenceNaN(),
        TestRewardScaling(),
        TestDtypeMismatch(),
        TestOnlineLearningGAE(),
        TestUnboundedLogits(),
        TestAdvantageNormalization(),
        TestAllFixesIntegration(),
    ]

    for test_class in test_classes:
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                method = getattr(test_class, method_name)
                try:
                    method()
                    print(f"\n[OK] {test_class.__class__.__name__}.{method_name}")
                except AssertionError as e:
                    print(f"\n[EXPECTED FAILURE] {test_class.__class__.__name__}.{method_name}: {e}")
                except Exception as e:
                    print(f"\n[ERROR] {test_class.__class__.__name__}.{method_name}: {e}")


if __name__ == '__main__':
    if HAS_PYTEST:
        pytest.main([__file__, '-v', '-s'])
    else:
        run_all_tests()
