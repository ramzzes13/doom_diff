"""Data collection from ViZDoom using trained PPO agent and random policy."""

import os
import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path

from .doom_env import DoomEnvironment
from .ppo_agent import PPOActorCritic, preprocess_obs


class TrajectoryCollector:
    """Collects gameplay trajectories from ViZDoom for diffusion model training."""

    def __init__(
        self,
        env: DoomEnvironment,
        save_dir: str = "data/trajectories",
        device: torch.device = torch.device("cpu"),
    ):
        self.env = env
        self.save_dir = save_dir
        self.device = device
        os.makedirs(save_dir, exist_ok=True)

    def collect_with_agent(
        self,
        agent: PPOActorCritic,
        num_trajectories: int = 100,
        max_length: int = 2048,
        prefix: str = "agent",
    ) -> List[str]:
        """Collect trajectories using a trained PPO agent."""
        agent.eval()
        saved_paths = []

        for traj_idx in range(num_trajectories):
            frames = []
            actions = []
            game_variables = []

            obs, info = self.env.reset()
            frames.append(obs.copy())
            game_variables.append(info.get("game_variables", {}))

            for step in range(max_length):
                obs_processed = preprocess_obs(obs)
                obs_tensor = torch.FloatTensor(obs_processed).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits, _ = agent(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()

                next_obs, reward, done, info = self.env.step(action)

                actions.append(action)

                if done:
                    # Reset and continue collecting
                    obs, info = self.env.reset()
                    frames.append(obs.copy())
                    game_variables.append(info.get("game_variables", {}))
                else:
                    obs = next_obs
                    frames.append(obs.copy())
                    game_variables.append(info.get("game_variables", {}))

            # Save trajectory
            path = self._save_trajectory(
                frames, actions, game_variables,
                f"{prefix}_{traj_idx:04d}"
            )
            saved_paths.append(path)

            if (traj_idx + 1) % 10 == 0:
                print(f"Collected {traj_idx + 1}/{num_trajectories} trajectories ({prefix})")

        return saved_paths

    def collect_random(
        self,
        num_trajectories: int = 50,
        max_length: int = 2048,
    ) -> List[str]:
        """Collect trajectories using a random policy."""
        saved_paths = []

        for traj_idx in range(num_trajectories):
            frames = []
            actions = []
            game_variables = []

            obs, info = self.env.reset()
            frames.append(obs.copy())
            game_variables.append(info.get("game_variables", {}))

            for step in range(max_length):
                action = np.random.randint(0, self.env.num_actions)
                next_obs, reward, done, info = self.env.step(action)

                actions.append(action)

                if done:
                    obs, info = self.env.reset()
                    frames.append(obs.copy())
                    game_variables.append(info.get("game_variables", {}))
                else:
                    obs = next_obs
                    frames.append(obs.copy())
                    game_variables.append(info.get("game_variables", {}))

            path = self._save_trajectory(
                frames, actions, game_variables,
                f"random_{traj_idx:04d}"
            )
            saved_paths.append(path)

            if (traj_idx + 1) % 10 == 0:
                print(f"Collected {traj_idx + 1}/{num_trajectories} random trajectories")

        return saved_paths

    def _save_trajectory(
        self,
        frames: List[np.ndarray],
        actions: List[int],
        game_variables: List[Dict],
        name: str,
    ) -> str:
        """Save a trajectory to disk as compressed npz."""
        path = os.path.join(self.save_dir, f"{name}.npz")

        # Stack frames
        frames_array = np.stack(frames, axis=0)  # (T+1, H, W, 3)
        actions_array = np.array(actions, dtype=np.int64)  # (T,)

        # Convert game variables to arrays
        var_names = sorted(game_variables[0].keys()) if game_variables[0] else []
        var_arrays = {}
        for var_name in var_names:
            var_arrays[f"var_{var_name}"] = np.array(
                [gv.get(var_name, 0.0) for gv in game_variables],
                dtype=np.float32
            )

        np.savez_compressed(
            path,
            frames=frames_array,
            actions=actions_array,
            var_names=np.array(var_names),
            **var_arrays,
        )

        return path


class DoomTrajectoryDataset(torch.utils.data.Dataset):
    """PyTorch dataset for loading DOOM trajectories for diffusion training."""

    def __init__(
        self,
        data_dir: str,
        num_context_frames: int = 4,
        resolution: tuple = (120, 160),  # H, W
        transform=None,
    ):
        self.data_dir = data_dir
        self.num_context_frames = num_context_frames
        self.resolution = resolution
        self.transform = transform

        # Find all trajectory files
        self.trajectory_files = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
        )

        # Build index: (file_idx, frame_idx) pairs
        self.index = []
        self.trajectory_cache = {}
        self._build_index()

    def _build_index(self):
        """Build an index of valid (trajectory, frame) pairs."""
        for file_idx, fpath in enumerate(self.trajectory_files):
            try:
                data = np.load(fpath, allow_pickle=True)
                num_frames = len(data["frames"])
                num_actions = len(data["actions"])
                data.close()

                # Need at least num_context_frames + 1 frames (context + target)
                min_start = self.num_context_frames
                for frame_idx in range(min_start, min(num_frames, num_actions + 1)):
                    self.index.append((file_idx, frame_idx))
            except Exception as e:
                print(f"Warning: Could not load {fpath}: {e}")

        print(f"Built dataset index: {len(self.index)} samples from {len(self.trajectory_files)} trajectories")

    def _load_trajectory(self, file_idx: int):
        """Load and cache a trajectory."""
        if file_idx not in self.trajectory_cache:
            # Keep cache small
            if len(self.trajectory_cache) > 50:
                oldest = next(iter(self.trajectory_cache))
                del self.trajectory_cache[oldest]

            data = np.load(self.trajectory_files[file_idx], allow_pickle=True)
            self.trajectory_cache[file_idx] = {
                "frames": data["frames"],
                "actions": data["actions"],
                "var_names": data["var_names"] if "var_names" in data else np.array([]),
            }
            # Load game variables
            var_names = self.trajectory_cache[file_idx]["var_names"]
            for var_name in var_names:
                key = f"var_{var_name}"
                if key in data:
                    self.trajectory_cache[file_idx][key] = data[key]
            data.close()

        return self.trajectory_cache[file_idx]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, frame_idx = self.index[idx]
        traj = self._load_trajectory(file_idx)

        # Context frames: [frame_idx - num_context_frames, ..., frame_idx - 1]
        # Target frame: frame_idx
        context_start = frame_idx - self.num_context_frames
        context_frames = traj["frames"][context_start:frame_idx]  # (N, H, W, 3)
        target_frame = traj["frames"][frame_idx]  # (H, W, 3)

        # Actions for the context period
        # Actions[i] transitions frames[i] -> frames[i+1]
        context_actions = traj["actions"][context_start:frame_idx]  # (N,)

        # Normalize frames to [-1, 1]
        context_frames = context_frames.astype(np.float32) / 127.5 - 1.0
        target_frame = target_frame.astype(np.float32) / 127.5 - 1.0

        # HWC -> CHW
        context_frames = np.transpose(context_frames, (0, 3, 1, 2))  # (N, 3, H, W)
        target_frame = np.transpose(target_frame, (2, 0, 1))  # (3, H, W)

        # Game variables for target frame
        game_vars = []
        var_names = traj["var_names"]
        for var_name in var_names:
            key = f"var_{var_name}"
            if key in traj and frame_idx < len(traj[key]):
                game_vars.append(traj[key][frame_idx])
            else:
                game_vars.append(0.0)
        game_vars = np.array(game_vars, dtype=np.float32)

        return {
            "context_frames": torch.FloatTensor(context_frames),  # (N, 3, H, W)
            "target_frame": torch.FloatTensor(target_frame),  # (3, H, W)
            "context_actions": torch.LongTensor(context_actions),  # (N,)
            "game_variables": torch.FloatTensor(game_vars),  # (num_vars,)
        }
