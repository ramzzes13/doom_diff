"""Step 4: Run full evaluation of the trained MemGameNGen model."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from memgamengen.doom_env import DoomEnvironment
from memgamengen.data_collector import DoomTrajectoryDataset
from memgamengen.diffusion_model import MemGameNGen
from memgamengen.trainer import MemGameNGenTrainer
from memgamengen.evaluation import MemGameNGenEvaluator


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("Creating MemGameNGen model...")
    model = MemGameNGen(
        pretrained_model="CompVis/stable-diffusion-v1-4",
        num_actions=8,
        num_context_frames=4,
        memory_enabled=True,
        num_memory_tokens=16,
        memory_dim=768,
        memory_update_type="gru",
        action_embed_dim=128,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        state_head_enabled=True,
        num_state_variables=2,
        noise_augmentation=True,
        noise_aug_max_level=0.7,
    )

    # Load checkpoint
    checkpoint_dir = os.path.join(project_dir, "checkpoints", "diffusion")
    # Find latest checkpoint
    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")
    ]) if os.path.exists(checkpoint_dir) else []

    if checkpoints:
        # Prefer final checkpoint
        ckpt = "memgamengen_final.pt" if "memgamengen_final.pt" in checkpoints else checkpoints[-1]
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        print(f"Loading checkpoint: {ckpt_path}")

        # Use trainer to load (handles LoRA properly)
        dummy_dataset = DoomTrajectoryDataset(
            data_dir=os.path.join(project_dir, "data", "trajectories"),
            num_context_frames=4,
        )
        trainer = MemGameNGenTrainer(
            model=model,
            train_dataset=dummy_dataset,
            device=device,
            max_train_steps=1,
        )
        trainer.load_checkpoint(ckpt_path)
        model = trainer.model
    else:
        print("WARNING: No checkpoint found. Evaluating untrained model.")

    model = model.to(device)
    model.eval()

    # Create environment
    env = DoomEnvironment(
        scenario="basic",
        resolution=(160, 120),
        frame_skip=4,
        visible=False,
        game_variables=["HEALTH", "AMMO2"],
    )

    # Create evaluation dataset
    data_dir = os.path.join(project_dir, "data", "trajectories")
    eval_dataset = DoomTrajectoryDataset(
        data_dir=data_dir,
        num_context_frames=4,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # Run evaluation
    evaluator = MemGameNGenEvaluator(
        model=model,
        env=env,
        device=device,
        results_dir=os.path.join(project_dir, "results"),
    )

    results = evaluator.run_full_evaluation(
        dataloader=eval_loader,
        num_autoreg_trajectories=10,
        autoreg_length=128,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for section, section_results in results.items():
        print(f"\n{section}:")
        if isinstance(section_results, dict):
            for k, v in section_results.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

    env.close()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
