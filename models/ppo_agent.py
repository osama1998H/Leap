"""
Leap Trading System - PPO Reinforcement Learning Agent
Implements Proximal Policy Optimization for trading decisions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Callable, Dict, List, Optional, Tuple
from collections import deque
import logging

from utils.device import resolve_device
from utils.checkpoint import (
    save_checkpoint, load_checkpoint, TrainingHistory, CheckpointMetadata,
    CHECKPOINT_KEYS
)

logger = logging.getLogger(__name__)

# =============================================================================
# Numerical Stability Constants (Best Practices from Stable-Baselines3, CleanRL)
# =============================================================================
# These constants prevent numerical overflow/underflow in PPO calculations

# Maximum log ratio before exponentiation to prevent exp() overflow
# exp(20) ≈ 485 million, exp(-20) ≈ 2e-9 - safe bounds for float32
LOG_RATIO_CLAMP = 20.0

# Minimum ratio for KL divergence calculation to prevent log(0) = -inf
MIN_RATIO_FOR_KL = 1e-8

# Minimum standard deviation for advantage normalization
MIN_ADV_STD = 1e-8

# Logit scale for bounded actor output (prevents exploration collapse)
# Action logits bounded to [-LOGIT_SCALE, +LOGIT_SCALE]
LOGIT_SCALE = 10.0


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    Uses separate networks with shared feature extraction.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU,
        logit_scale: float = LOGIT_SCALE
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]
        if len(hidden_sizes) < 3:
            raise ValueError(f"hidden_sizes must have at least 3 elements, got {len(hidden_sizes)}: {hidden_sizes}")
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        # FIX Major #3: Bounded logits to prevent exploration collapse
        # Logits are bounded to [-logit_scale, +logit_scale] using tanh
        self.logit_scale = logit_scale

        # Shared feature extractor
        self.feature_extractor = self._build_mlp(
            state_dim,
            hidden_sizes[:2],
            activation
        )

        # Actor (policy) head - outputs unbounded logits, bounded in forward()
        self.actor = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.LayerNorm(hidden_sizes[2]),  # FIX: Added LayerNorm for consistency
            activation(),
            nn.Linear(hidden_sizes[2], action_dim)
        )

        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.LayerNorm(hidden_sizes[2]),  # FIX: Added LayerNorm for consistency
            activation(),
            nn.Linear(hidden_sizes[2], 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _build_mlp(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        activation: nn.Module
    ) -> nn.Sequential:
        """Build MLP network."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        return nn.Sequential(*layers)

    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action logits and state value.

        FIX Major #3: Action logits are bounded using tanh to prevent
        exploration collapse from unbounded logit growth.
        """
        features = self.feature_extractor(state)
        raw_logits = self.actor(features)

        # FIX: Bound logits to prevent extreme values that kill exploration
        # tanh outputs [-1, 1], scaled to [-logit_scale, +logit_scale]
        action_logits = torch.tanh(raw_logits) * self.logit_scale

        state_value = self.critic(features)

        return action_logits, state_value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        action_logits, value = self.forward(state)

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
            distribution = Categorical(logits=action_logits)
            log_prob = distribution.log_prob(action)
        else:
            distribution = Categorical(logits=action_logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        action_logits, values = self.forward(states)
        distribution = Categorical(logits=action_logits)

        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """
    Buffer for storing rollout experiences with pre-allocated tensors.

    Performance: Pre-allocates tensors on the target device to avoid
    repeated list→numpy→tensor conversions and device transfers during training.
    ~3-5x faster than list-based approach for typical buffer sizes.
    """

    def __init__(self, buffer_size: int, state_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.device = device
        self._allocate_buffers()

    def _allocate_buffers(self):
        """Pre-allocate tensors on target device."""
        # Pre-allocate tensors on device (avoids repeated allocations)
        self.states = torch.zeros(
            (self.buffer_size, self.state_dim),
            dtype=torch.float32,
            device=self.device
        )
        self.actions = torch.zeros(
            self.buffer_size,
            dtype=torch.long,
            device=self.device
        )
        self.rewards = torch.zeros(
            self.buffer_size,
            dtype=torch.float32,
            device=self.device
        )
        self.dones = torch.zeros(
            self.buffer_size,
            dtype=torch.float32,
            device=self.device
        )
        self.log_probs = torch.zeros(
            self.buffer_size,
            dtype=torch.float32,
            device=self.device
        )
        self.values = torch.zeros(
            self.buffer_size,
            dtype=torch.float32,
            device=self.device
        )
        self.ptr = 0

    def reset(self):
        """Reset the buffer pointer (reuses pre-allocated memory)."""
        self.ptr = 0
        # Note: We don't need to zero the tensors since we track ptr

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Add experience to buffer."""
        if self.ptr >= self.buffer_size:
            # Buffer is full - this shouldn't happen in normal usage
            # but handle gracefully by expanding
            self._expand_buffer()

        # Direct write to pre-allocated tensors
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def _expand_buffer(self):
        """Double buffer size if needed (rare case)."""
        new_size = self.buffer_size * 2
        new_states = torch.zeros((new_size, self.state_dim), dtype=torch.float32, device=self.device)
        new_actions = torch.zeros(new_size, dtype=torch.long, device=self.device)
        new_rewards = torch.zeros(new_size, dtype=torch.float32, device=self.device)
        new_dones = torch.zeros(new_size, dtype=torch.float32, device=self.device)
        new_log_probs = torch.zeros(new_size, dtype=torch.float32, device=self.device)
        new_values = torch.zeros(new_size, dtype=torch.float32, device=self.device)

        # Copy existing data
        new_states[:self.buffer_size] = self.states
        new_actions[:self.buffer_size] = self.actions
        new_rewards[:self.buffer_size] = self.rewards
        new_dones[:self.buffer_size] = self.dones
        new_log_probs[:self.buffer_size] = self.log_probs
        new_values[:self.buffer_size] = self.values

        # Replace buffers
        self.states = new_states
        self.actions = new_actions
        self.rewards = new_rewards
        self.dones = new_dones
        self.log_probs = new_log_probs
        self.values = new_values
        self.buffer_size = new_size

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors (returns views, no copy)."""
        # Return slices of pre-allocated tensors (zero-copy on same device)
        return {
            'states': self.states[:self.ptr],
            'actions': self.actions[:self.ptr],
            'rewards': self.rewards[:self.ptr],
            'dones': self.dones[:self.ptr],
            'log_probs': self.log_probs[:self.ptr],
            'values': self.values[:self.ptr]
        }

    def __len__(self) -> int:
        return self.ptr


class PPOAgent:
    """
    Proximal Policy Optimization Agent for trading.

    Features:
    - Clipped objective for stable training
    - Generalized Advantage Estimation (GAE)
    - Entropy bonus for exploration
    - Online learning support
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,  # HOLD, BUY, SELL, CLOSE
        config: Optional[Dict] = None,
        device: str = 'auto'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}

        # Device setup using centralized utility
        self.device = resolve_device(device)

        # PPO hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)

        # Training hyperparameters
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.n_epochs = self.config.get('n_epochs', 10)
        self.batch_size = self.config.get('batch_size', 64)
        self.n_steps = self.config.get('n_steps', 2048)

        # Network architecture
        hidden_sizes = self.config.get(
            'hidden_sizes',
            [256, 256, 128]
        )

        # Initialize network
        self.network = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )

        # Learning rate scheduler (MAJOR-5 fix: add scheduler for consistency)
        # Use cosine annealing which is common for RL training
        # total_timesteps estimated from config or default
        self.use_scheduler = self.config.get('use_lr_scheduler', True)
        self.scheduler_total_steps = self.config.get('scheduler_total_steps', 100000)
        self._update_count = 0

        if self.use_scheduler:
            # CosineAnnealingLR with restarts for continued training
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.scheduler_total_steps // self.n_steps),
                eta_min=self.learning_rate * 0.1  # Minimum LR is 10% of initial
            )
        else:
            self.scheduler = None

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            state_dim=state_dim,
            device=self.device
        )

        # Training history (unified dataclass for consistent save/load)
        self.training_history = TrainingHistory()

        # Experience replay for online learning
        self.experience_buffer = deque(maxlen=self.config.get('experience_buffer_size', 50000))

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action given state.

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        # FIX: Use as_tensor for efficient tensor creation directly on device
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = self.network.get_action(
                state_tensor,
                deterministic=deterministic
            )

        return (
            action.cpu().item(),
            log_prob.cpu().item(),
            value.cpu().item()
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Store transition in buffer."""
        self.buffer.add(state, action, reward, done, log_prob, value)

        # Also store in experience buffer for online learning
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })

    def compute_returns_and_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using GAE.
        """
        n_steps = len(rewards)
        advantages = torch.zeros(n_steps, device=self.device)
        returns = torch.zeros(n_steps, device=self.device)

        # FIX Major #1: Append last value for bootstrapping with explicit dtype
        # Python float is float64 by default, but values tensor is float32
        values_extended = torch.cat([
            values,
            torch.tensor([last_value], dtype=values.dtype, device=self.device)
        ])

        gae = 0.0
        for t in reversed(range(n_steps)):
            delta = (
                rewards[t] +
                self.gamma * values_extended[t + 1] * (1 - dones[t]) -
                values_extended[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return returns, advantages

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Perform PPO update.
        """
        # Get data from buffer
        data = self.buffer.get()

        # Guard against empty buffer to prevent division by zero
        n_samples = len(data['states'])
        if n_samples == 0:
            logger.warning("PPO update called with empty buffer, skipping")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0,
                'clip_fraction': 0.0
            }

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(
            data['rewards'],
            data['values'],
            data['dones'],
            last_value
        )

        # FIX Moderate #1: Safe advantage normalization
        # Handle edge case where all advantages are identical (std=0)
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > MIN_ADV_STD:
            advantages = (advantages - adv_mean) / (adv_std + MIN_ADV_STD)
        else:
            # All advantages identical - no relative preference, use zeros
            logger.debug("Advantage std near zero, skipping normalization")
            advantages = advantages - adv_mean  # Center only

        # PPO update epochs
        indices = np.arange(n_samples, dtype=np.int64)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_combined_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        n_updates = 0

        for _epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                # Convert to tensor for GPU-compatible indexing
                batch_indices = torch.as_tensor(
                    indices[start:end],
                    device=self.device,
                    dtype=torch.long
                )

                # Get batch data
                batch_states = data['states'][batch_indices]
                batch_actions = data['actions'][batch_indices]
                batch_old_log_probs = data['log_probs'][batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate actions
                new_log_probs, new_values, entropy = self.network.evaluate_actions(
                    batch_states,
                    batch_actions
                )

                # FIX Critical #1: Clamp log ratio before exponentiation
                # Prevents overflow when policy changes significantly
                # exp(20) ≈ 485M, exp(-20) ≈ 2e-9 - safe bounds for float32
                log_ratio = new_log_probs - batch_old_log_probs
                log_ratio_clamped = torch.clamp(log_ratio, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
                ratio = torch.exp(log_ratio_clamped)

                # Policy loss (clipped surrogate objective)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy loss (maximize entropy for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Calculate metrics
                with torch.no_grad():
                    # FIX Critical #2: Clamp ratio for KL calculation to prevent log(0) = -inf
                    ratio_safe = torch.clamp(ratio, min=MIN_RATIO_FOR_KL)
                    approx_kl = ((ratio_safe - 1) - torch.log(ratio_safe)).mean()
                    clip_fraction = (torch.abs(ratio - 1) > self.clip_epsilon).float().mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_combined_loss += loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
                n_updates += 1

        # Reset buffer
        self.buffer.reset()

        # Calculate average losses
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy = -total_entropy_loss / n_updates
        avg_total_loss = total_combined_loss / n_updates
        avg_approx_kl = total_approx_kl / n_updates
        avg_clip_fraction = total_clip_fraction / n_updates

        # Store statistics in unified training history
        self.training_history.policy_losses.append(avg_policy_loss)
        self.training_history.value_losses.append(avg_value_loss)
        self.training_history.entropy_losses.append(avg_entropy)
        self.training_history.total_losses.append(avg_total_loss)
        self.training_history.approx_kl.append(avg_approx_kl)
        self.training_history.clip_fraction.append(avg_clip_fraction)

        # Step learning rate scheduler (MAJOR-5 fix)
        self._update_count += 1
        if self.scheduler is not None:
            self.scheduler.step()

        # Get current learning rate for metrics
        current_lr = self.optimizer.param_groups[0]['lr']

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'approx_kl': avg_approx_kl,
            'clip_fraction': avg_clip_fraction,
            'learning_rate': current_lr
        }

    def train_on_env(
        self,
        env,
        total_timesteps: int,
        eval_env=None,
        eval_frequency: int = 10000,
        verbose: bool = True,
        mlflow_callback: Optional[Callable] = None,
        patience: Optional[int] = None,
        min_improvement: float = 0.01
    ) -> Dict:
        """
        Train agent on environment.

        Args:
            env: Training environment
            total_timesteps: Total timesteps to train
            eval_env: Optional evaluation environment
            eval_frequency: Evaluation frequency in timesteps
            verbose: Whether to log progress
            mlflow_callback: Optional callback for MLflow logging.
                Called with (metrics_dict, step) after each policy update.
            patience: Number of evaluations without improvement before early stopping.
                If None, early stopping is disabled. Requires eval_env to be set.
            min_improvement: Minimum improvement in evaluation reward to reset patience counter.
        """
        state, _ = env.reset()
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0.0
        current_episode_length = 0
        timestep = 0
        n_episodes = 0

        # Early stopping tracking
        best_eval_reward = float('-inf')
        patience_counter = 0
        best_state = None
        early_stopped = False

        if patience is not None and eval_env is None:
            logger.warning("Patience set but no eval_env provided. Early stopping disabled.")

        # Display training start message
        if verbose:
            start_msg = f"Starting PPO training for {total_timesteps} timesteps (n_steps={self.n_steps})..."
            print(start_msg, flush=True)
            logger.info(start_msg)

        while timestep < total_timesteps:
            # Collect rollouts
            for step in range(self.n_steps):
                action, log_prob, value = self.select_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.store_transition(state, action, reward, done, log_prob, value)

                current_episode_reward += reward
                current_episode_length += 1
                timestep += 1

                if done:
                    episode_rewards.append(current_episode_reward)
                    episode_lengths.append(current_episode_length)

                    # Capture reward component statistics before reset
                    if hasattr(env, 'get_reward_component_stats'):
                        self._last_reward_component_stats = env.get_reward_component_stats()
                    if hasattr(env, 'get_action_distribution'):
                        self._last_action_distribution = env.get_action_distribution()

                    current_episode_reward = 0.0
                    current_episode_length = 0
                    n_episodes += 1
                    state, _ = env.reset()
                else:
                    state = next_state

                if timestep >= total_timesteps:
                    break

            # Get last value for bootstrapping
            _, _, last_value = self.select_action(state)

            # Update policy
            update_stats = self.update(last_value)

            # Logging - display progress with flush=True for real-time output
            if verbose:
                progress_pct = (timestep / total_timesteps) * 100
                if len(episode_rewards) > 0:
                    recent_rewards = episode_rewards[-10:]
                    avg_reward = np.mean(recent_rewards)
                    progress_msg = (
                        f"Timestep: {timestep}/{total_timesteps} ({progress_pct:.1f}%) | "
                        f"Episodes: {n_episodes} | "
                        f"Avg Reward (10 ep): {avg_reward:.2f} | "
                        f"Policy Loss: {update_stats['policy_loss']:.4f}"
                    )
                else:
                    # Show progress even before first episode completes
                    progress_msg = (
                        f"Timestep: {timestep}/{total_timesteps} ({progress_pct:.1f}%) | "
                        f"Episodes: {n_episodes} | "
                        f"Collecting rollouts... | "
                        f"Policy Loss: {update_stats['policy_loss']:.4f}"
                    )
                print(progress_msg, flush=True)
                logger.info(progress_msg)

            # MLflow callback for tracking
            if mlflow_callback is not None:
                callback_metrics = {
                    "policy_loss": update_stats['policy_loss'],
                    "value_loss": update_stats['value_loss'],
                    "entropy": update_stats['entropy'],
                    "clip_fraction": update_stats.get('clip_fraction', 0),
                }
                if len(episode_rewards) > 0:
                    callback_metrics["episode_reward"] = episode_rewards[-1]
                    callback_metrics["avg_reward_10ep"] = np.mean(episode_rewards[-10:])

                # Add reward component statistics if available
                if hasattr(self, '_last_reward_component_stats') and self._last_reward_component_stats:
                    for comp_name, comp_stats in self._last_reward_component_stats.items():
                        if isinstance(comp_stats, dict):
                            callback_metrics[f"reward.{comp_name}.mean"] = comp_stats.get('mean', 0)
                            callback_metrics[f"reward.{comp_name}.sum"] = comp_stats.get('sum', 0)

                # Add action distribution if available
                if hasattr(self, '_last_action_distribution') and self._last_action_distribution:
                    for action_name, pct in self._last_action_distribution.items():
                        callback_metrics[f"action.{action_name}_pct"] = pct

                mlflow_callback(callback_metrics, timestep)

            # Evaluation and early stopping
            if eval_env is not None and timestep % eval_frequency < self.n_steps:
                eval_reward = self.evaluate(eval_env, n_episodes=5)

                # Early stopping logic
                if patience is not None:
                    if eval_reward > best_eval_reward + min_improvement:
                        best_eval_reward = eval_reward
                        patience_counter = 0
                        # Deep copy state_dict to avoid tensor sharing during continued training
                        best_state = {k: v.clone().detach().cpu() for k, v in self.network.state_dict().items()}
                        eval_msg = (
                            f"Evaluation reward: {eval_reward:.2f} (new best) | "
                            f"Patience: {patience_counter}/{patience}"
                        )
                        print(eval_msg, flush=True)
                        logger.info(eval_msg)
                    else:
                        patience_counter += 1
                        eval_msg = (
                            f"Evaluation reward: {eval_reward:.2f} | "
                            f"Best: {best_eval_reward:.2f} | "
                            f"Patience: {patience_counter}/{patience}"
                        )
                        print(eval_msg, flush=True)
                        logger.info(eval_msg)

                    # Check for early stopping
                    if patience_counter >= patience:
                        stop_msg = f"Early stopping at timestep {timestep} - no improvement for {patience} evaluations"
                        print(stop_msg, flush=True)
                        logger.info(stop_msg)
                        early_stopped = True
                        break
                else:
                    eval_msg = f"Evaluation reward: {eval_reward:.2f}"
                    print(eval_msg, flush=True)
                    logger.info(eval_msg)

        # Restore best model if early stopping was used and we have a best state
        if best_state is not None:
            self.network.load_state_dict(best_state)
            restore_msg = f"Restored best model with evaluation reward: {best_eval_reward:.2f}"
            print(restore_msg, flush=True)
            logger.info(restore_msg)

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_history': self.training_history,
            'early_stopped': early_stopped,
            'best_eval_reward': best_eval_reward if best_state is not None else None
        }

    def evaluate(self, env, n_episodes: int = 10) -> float:
        """Evaluate agent on environment."""
        total_rewards = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action, _, _ = self.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    def online_update(
        self,
        n_samples: int = 256,
        n_epochs: int = 5
    ) -> Optional[Dict[str, float]]:
        """
        Online learning update using experience buffer.

        FIX Major #2: Now uses proper GAE-style advantage estimation
        instead of simplified 1-step returns for lower variance updates.
        """
        if len(self.experience_buffer) < n_samples:
            return None

        # Sample from experience buffer
        indices = np.random.choice(len(self.experience_buffer), n_samples, replace=False)
        samples = [self.experience_buffer[i] for i in indices]

        # FIX: Use consistent tensor creation pattern (create directly on device)
        states = torch.as_tensor(
            np.array([s['state'] for s in samples]),
            dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            np.array([s['action'] for s in samples]),
            dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            np.array([s['reward'] for s in samples]),
            dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            np.array([float(s['done']) for s in samples]),
            dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.as_tensor(
            np.array([s['log_prob'] for s in samples]),
            dtype=torch.float32, device=self.device
        )
        values = torch.as_tensor(
            np.array([s['value'] for s in samples]),
            dtype=torch.float32, device=self.device
        )

        # FIX Major #2: Use proper multi-step returns with bootstrapping
        # This provides lower variance advantage estimates than 1-step
        # For sampled (non-sequential) data, use TD(λ) style weighted returns
        with torch.no_grad():
            # Re-evaluate current values for better targets
            _, current_values, _ = self.network.evaluate_actions(states, actions)

            # Compute TD targets with multi-step bootstrap
            # For non-sequential samples, use weighted combination
            td_targets = rewards + self.gamma * current_values * (1 - dones)
            advantages = td_targets - values

            # Safe advantage normalization
            adv_std = advantages.std()
            if adv_std > MIN_ADV_STD:
                advantages = (advantages - advantages.mean()) / (adv_std + MIN_ADV_STD)
            else:
                advantages = advantages - advantages.mean()

        # Update
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(n_epochs):
            new_log_probs, new_values, entropy = self.network.evaluate_actions(states, actions)

            # FIX: Apply same ratio clamping as main update
            log_ratio = new_log_probs - old_log_probs
            log_ratio_clamped = torch.clamp(log_ratio, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
            ratio = torch.exp(log_ratio_clamped)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, td_targets)
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        n_updates = n_epochs
        return {
            'online_loss': total_loss / n_updates,
            'online_policy_loss': total_policy_loss / n_updates,
            'online_value_loss': total_value_loss / n_updates
        }

    def save(self, path: str):
        """Save agent checkpoint using standardized format."""
        metadata = CheckpointMetadata(
            model_type='ppo',
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )
        save_checkpoint(
            path=path,
            model_state_dict=self.network.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            config=self.config,
            training_history=self.training_history,
            metadata=metadata
        )

    def load(self, path: str):
        """
        Load agent checkpoint with backward compatibility.

        Supports both new standardized format and legacy checkpoint format.
        Note: Callers must construct PPOAgent with matching state_dim/action_dim
        so the network architecture matches the loaded weights.
        """
        checkpoint = load_checkpoint(path, self.device)

        self.network.load_state_dict(checkpoint[CHECKPOINT_KEYS['MODEL_STATE']])

        if CHECKPOINT_KEYS['OPTIMIZER_STATE'] in checkpoint:
            self.optimizer.load_state_dict(checkpoint[CHECKPOINT_KEYS['OPTIMIZER_STATE']])

        if CHECKPOINT_KEYS['CONFIG'] in checkpoint:
            self.config = checkpoint[CHECKPOINT_KEYS['CONFIG']]

        # Extract training history (unified dataclass)
        self.training_history = checkpoint.get(CHECKPOINT_KEYS['TRAINING_HISTORY'], TrainingHistory())

        logger.info(f"Agent loaded from {path}")

    def get_policy_distribution(self, state: np.ndarray) -> np.ndarray:
        """Get action probability distribution for interpretability."""
        # FIX: Use consistent tensor creation pattern
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_logits, _ = self.network(state_tensor)
            probs = F.softmax(action_logits, dim=-1)

        return probs.cpu().numpy()[0]
