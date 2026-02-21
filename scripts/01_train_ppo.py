"""Step 1: Train PPO agent for DOOM data collection."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from memgamengen.doom_env import DoomEnvironment
from memgamengen.ppo_agent import PPOTrainer


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    print("Creating ViZDoom environment...")
    env = DoomEnvironment(
        scenario="basic",
        resolution=(160, 120),
        frame_skip=4,
        visible=False,
        game_variables=["HEALTH", "AMMO2"],
    )
    print(f"Environment created. Actions: {env.num_actions}")

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        device=device,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        steps_per_update=512,
        num_epochs=4,
        batch_size=128,
    )

    # Train
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "checkpoints", "ppo")
    trainer.train(
        total_timesteps=200000,
        checkpoint_dir=checkpoint_dir,
        log_interval=5,
    )

    # Save final model
    final_path = os.path.join(checkpoint_dir, "ppo_final.pt")
    trainer.save(final_path)
    print(f"PPO training complete. Final model: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
