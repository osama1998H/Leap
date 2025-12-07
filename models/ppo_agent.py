"""
Leap Trading System - PPO Reinforcement Learning Agent
Implements Proximal Policy Optimization for trading decisions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


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
        activation: nn.Module = nn.ReLU
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]
        if len(hidden_sizes) < 3:
            raise ValueError(f"hidden_sizes must have at least 3 elements, got {len(hidden_sizes)}: {hidden_sizes}")
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        self.feature_extractor = self._build_mlp(
            state_dim,
            hidden_sizes[:2],
            activation
        )

        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            activation(),
            nn.Linear(hidden_sizes[2], action_dim)
        )

        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
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
        """
        features = self.feature_extractor(state)
        action_logits = self.actor(features)
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
    Buffer for storing rollout experiences.
    """

    def __init__(self, buffer_size: int, state_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.device = device
        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.ptr = 0

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
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.ptr += 1

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors."""
        return {
            'states': torch.FloatTensor(np.array(self.states)).to(self.device),
            'actions': torch.LongTensor(np.array(self.actions)).to(self.device),
            'rewards': torch.FloatTensor(np.array(self.rewards)).to(self.device),
            'dones': torch.FloatTensor(np.array(self.dones)).to(self.device),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)).to(self.device),
            'values': torch.FloatTensor(np.array(self.values)).to(self.device)
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

        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

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

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            state_dim=state_dim,
            device=self.device
        )

        # Training statistics
        self.training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'total_losses': [],
            'approx_kl': [],
            'clip_fraction': []
        }

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
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

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

        # Append last value for bootstrapping
        values_extended = torch.cat([values, torch.tensor([last_value], device=self.device)])

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

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(
            data['rewards'],
            data['values'],
            data['dones'],
            last_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        n_samples = len(data['states'])
        indices = np.arange(n_samples, dtype=np.int64)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_combined_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
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

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy loss
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
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
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

        # Store statistics
        self.training_stats['policy_losses'].append(avg_policy_loss)
        self.training_stats['value_losses'].append(avg_value_loss)
        self.training_stats['entropy_losses'].append(avg_entropy)
        self.training_stats['total_losses'].append(avg_total_loss)
        self.training_stats['approx_kl'].append(avg_approx_kl)
        self.training_stats['clip_fraction'].append(avg_clip_fraction)

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'approx_kl': avg_approx_kl,
            'clip_fraction': avg_clip_fraction
        }

    def train_on_env(
        self,
        env,
        total_timesteps: int,
        eval_env=None,
        eval_frequency: int = 10000,
        verbose: bool = True
    ) -> Dict:
        """
        Train agent on environment.
        """
        state, _ = env.reset()
        episode_rewards = []
        current_episode_reward = 0.0
        timestep = 0
        n_episodes = 0

        while timestep < total_timesteps:
            # Collect rollouts
            for step in range(self.n_steps):
                action, log_prob, value = self.select_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.store_transition(state, action, reward, done, log_prob, value)

                current_episode_reward += reward
                timestep += 1

                if done:
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0.0
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

            # Logging
            if verbose and len(episode_rewards) > 0:
                recent_rewards = episode_rewards[-10:]
                avg_reward = np.mean(recent_rewards)
                logger.info(
                    f"Timestep: {timestep}/{total_timesteps} | "
                    f"Episodes: {n_episodes} | "
                    f"Avg Reward (10 ep): {avg_reward:.2f} | "
                    f"Policy Loss: {update_stats['policy_loss']:.4f}"
                )

            # Evaluation
            if eval_env is not None and timestep % eval_frequency < self.n_steps:
                eval_reward = self.evaluate(eval_env, n_episodes=5)
                logger.info(f"Evaluation reward: {eval_reward:.2f}")

        return {
            'episode_rewards': episode_rewards,
            'training_stats': self.training_stats
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
        """
        if len(self.experience_buffer) < n_samples:
            return None

        # Sample from experience buffer
        indices = np.random.choice(len(self.experience_buffer), n_samples, replace=False)
        samples = [self.experience_buffer[i] for i in indices]

        # Prepare data
        states = torch.FloatTensor([s['state'] for s in samples]).to(self.device)
        actions = torch.LongTensor([s['action'] for s in samples]).to(self.device)
        rewards = torch.FloatTensor([s['reward'] for s in samples]).to(self.device)
        dones = torch.FloatTensor([s['done'] for s in samples]).to(self.device)
        old_log_probs = torch.FloatTensor([s['log_prob'] for s in samples]).to(self.device)
        values = torch.FloatTensor([s['value'] for s in samples]).to(self.device)

        # Compute returns (simplified for online learning)
        returns = rewards + self.gamma * values * (1 - dones)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update
        total_loss = 0.0

        for _ in range(n_epochs):
            new_log_probs, new_values, entropy = self.network.evaluate_actions(states, actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        return {'online_loss': total_loss / n_epochs}

    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, path)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        """
        Load agent checkpoint.

        Note: Callers must construct PPOAgent with matching state_dim/action_dim
        so the network architecture matches the loaded weights.
        """
        # weights_only=False required for loading optimizer state and custom objects
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint.get('config', self.config)
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        logger.info(f"Agent loaded from {path}")

    def get_policy_distribution(self, state: np.ndarray) -> np.ndarray:
        """Get action probability distribution for interpretability."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, _ = self.network(state_tensor)
            probs = F.softmax(action_logits, dim=-1)

        return probs.cpu().numpy()[0]
