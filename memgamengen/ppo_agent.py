"""PPO agent for collecting DOOM gameplay trajectories."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import os


class CNNFeatureExtractor(nn.Module):
    """Simple CNN feature extractor for DOOM frames."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        # Input: (B, 3, 120, 160)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4, padding=2),  # -> (B, 32, 30, 40)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # -> (B, 64, 15, 20)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # -> (B, 64, 15, 20)
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 15 * 20, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) normalized to [0, 1]
        features = self.conv(x)
        features = features.reshape(features.size(0), -1)
        return self.fc(features)


class PPOActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, num_actions: int = 8, feature_dim: int = 256):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(feature_dim=feature_dim)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value.squeeze(-1)

    def get_action(self, obs: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate log probs and values for given obs-action pairs."""
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


class RolloutBuffer:
    """Buffer for storing PPO rollout data."""

    def __init__(self, buffer_size: int, obs_shape: Tuple, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.obs = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, log_prob, value):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE advantages and returns."""
        last_gae = 0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_batches(self, batch_size: int):
        """Yield mini-batches for PPO updates."""
        indices = np.random.permutation(self.ptr)
        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            batch_indices = indices[start:end]

            yield {
                "obs": torch.FloatTensor(self.obs[batch_indices]).to(self.device),
                "actions": torch.LongTensor(self.actions[batch_indices]).to(self.device),
                "log_probs": torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                "advantages": torch.FloatTensor(self.advantages[batch_indices]).to(self.device),
                "returns": torch.FloatTensor(self.returns[batch_indices]).to(self.device),
            }

    def reset(self):
        self.ptr = 0


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """Convert HWC uint8 observation to CHW float32 [0, 1]."""
    obs = obs.astype(np.float32) / 255.0
    obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
    return obs


class PPOTrainer:
    """PPO trainer for the DOOM agent."""

    def __init__(
        self,
        env,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        steps_per_update: int = 512,
        num_epochs: int = 4,
        batch_size: int = 128,
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.steps_per_update = steps_per_update
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.model = PPOActorCritic(
            num_actions=env.num_actions, feature_dim=256
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.obs_shape = (3, 120, 160)
        self.buffer = RolloutBuffer(steps_per_update, self.obs_shape, device)

    def collect_rollout(self) -> Dict:
        """Collect a rollout of experience."""
        self.buffer.reset()
        obs, info = self.env.reset()
        obs_processed = preprocess_obs(obs)

        total_reward = 0
        episodes_completed = 0

        for step in range(self.steps_per_update):
            obs_tensor = torch.FloatTensor(obs_processed).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_tensor)

            next_obs, reward, done, info = self.env.step(action)
            total_reward += reward

            self.buffer.add(
                obs_processed, action, reward, float(done),
                log_prob.item(), value.item()
            )

            if done:
                obs, info = self.env.reset()
                obs_processed = preprocess_obs(obs)
                episodes_completed += 1
            else:
                obs = next_obs
                obs_processed = preprocess_obs(obs)

        # Compute last value for GAE
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_processed).unsqueeze(0).to(self.device)
            _, last_value = self.model(obs_tensor)
            last_value = last_value.item()

        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)

        return {
            "total_reward": total_reward,
            "episodes_completed": max(episodes_completed, 1),
            "avg_reward": total_reward / max(episodes_completed, 1),
        }

    def update(self) -> Dict:
        """Perform PPO update."""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.num_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                log_probs, values, entropy = self.model.evaluate_actions(
                    batch["obs"], batch["actions"]
                )

                # Normalize advantages
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss (clipped)
                ratio = torch.exp(log_probs - batch["log_probs"])
                policy_loss1 = -advantages * ratio
                policy_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch["returns"])

                # Entropy
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def train(self, total_timesteps: int, checkpoint_dir: str = "checkpoints/ppo",
              log_interval: int = 1) -> None:
        """Train the PPO agent."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        num_updates = total_timesteps // self.steps_per_update

        print(f"Training PPO agent for {total_timesteps} timesteps ({num_updates} updates)")

        for update_idx in range(num_updates):
            rollout_stats = self.collect_rollout()
            update_stats = self.update()

            timestep = (update_idx + 1) * self.steps_per_update

            if (update_idx + 1) % log_interval == 0:
                print(
                    f"Update {update_idx + 1}/{num_updates} | "
                    f"Timestep {timestep}/{total_timesteps} | "
                    f"Avg Reward: {rollout_stats['avg_reward']:.2f} | "
                    f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                    f"Value Loss: {update_stats['value_loss']:.4f} | "
                    f"Entropy: {update_stats['entropy']:.4f}"
                )

            # Save checkpoint
            if timestep % 50000 == 0 or update_idx == num_updates - 1:
                path = os.path.join(checkpoint_dir, f"ppo_step_{timestep}.pt")
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "timestep": timestep,
                }, path)
                print(f"Saved checkpoint: {path}")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
