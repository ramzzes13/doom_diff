"""Step 2: Collect gameplay trajectories using trained PPO agent + random policy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from memgamengen.doom_env import DoomEnvironment
from memgamengen.ppo_agent import PPOActorCritic
from memgamengen.data_collector import TrajectoryCollector


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create environment
    print("Creating ViZDoom environment...")
    env = DoomEnvironment(
        scenario="basic",
        resolution=(160, 120),
        frame_skip=4,
        visible=False,
        game_variables=["HEALTH", "AMMO2"],
    )

    # Load trained PPO agent
    agent = PPOActorCritic(num_actions=env.num_actions, feature_dim=256).to(device)
    ppo_path = os.path.join(project_dir, "checkpoints", "ppo", "ppo_final.pt")
    if os.path.exists(ppo_path):
        agent.load_state_dict(torch.load(ppo_path, map_location=device, weights_only=True))
        print(f"Loaded PPO agent from {ppo_path}")
    else:
        print("WARNING: No trained PPO agent found. Using random initialization.")

    # Create collector
    save_dir = os.path.join(project_dir, "data", "trajectories")
    collector = TrajectoryCollector(env=env, save_dir=save_dir, device=device)

    # Collect agent trajectories
    print("\n=== Collecting agent trajectories ===")
    agent_paths = collector.collect_with_agent(
        agent=agent,
        num_trajectories=100,
        max_trajectory_length=1024,
        prefix="agent",
    )
    print(f"Collected {len(agent_paths)} agent trajectories")

    # Collect random trajectories
    print("\n=== Collecting random trajectories ===")
    random_paths = collector.collect_random(
        num_trajectories=20,
        max_trajectory_length=1024,
    )
    print(f"Collected {len(random_paths)} random trajectories")

    total = len(agent_paths) + len(random_paths)
    print(f"\nTotal trajectories: {total}")
    print(f"Saved to: {save_dir}")

    env.close()


if __name__ == "__main__":
    main()
