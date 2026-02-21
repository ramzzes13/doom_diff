"""Step 3: Train the MemGameNGen memory-augmented diffusion model."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from memgamengen.data_collector import DoomTrajectoryDataset
from memgamengen.diffusion_model import MemGameNGen
from memgamengen.trainer import MemGameNGenTrainer


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check GPU memory
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
        print(f"Free GPU memory: {free_mem:.1f} GB")

    # Load dataset
    data_dir = os.path.join(project_dir, "data", "trajectories")
    print(f"Loading dataset from {data_dir}...")

    full_dataset = DoomTrajectoryDataset(
        data_dir=data_dir,
        num_context_frames=4,
        resolution=(120, 160),
    )

    # Split into train/val
    total_len = len(full_dataset)
    val_len = max(1, int(total_len * 0.1))
    train_len = total_len - val_len
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_len}, Val: {val_len}")

    # Create model
    print("\nCreating MemGameNGen model...")
    model = MemGameNGen(
        pretrained_model="CompVis/stable-diffusion-v1-4",
        num_actions=8,
        num_context_frames=4,
        # Memory
        memory_enabled=True,
        num_memory_tokens=16,
        memory_dim=768,
        memory_update_type="gru",
        # Action
        action_embed_dim=128,
        # LoRA
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        # State head
        state_head_enabled=True,
        num_state_variables=2,  # HEALTH, AMMO2
        # Noise augmentation
        noise_augmentation=True,
        noise_aug_max_level=0.7,
    )

    print(f"Trainable parameters: {model.count_trainable_params():,}")

    # Create trainer
    trainer = MemGameNGenTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=1e-4,
        batch_size=1,
        gradient_accumulation_steps=4,
        max_train_steps=10000,
        warmup_steps=200,
        diffusion_loss_weight=1.0,
        state_loss_weight=0.1,
        save_dir=os.path.join(project_dir, "checkpoints", "diffusion"),
        log_dir=os.path.join(project_dir, "results", "logs"),
        save_every=2000,
        eval_every=1000,
        log_every=50,
        mixed_precision=True,
        gradient_checkpointing=True,
        device=device,
        seed=42,
    )

    # Train
    trainer.train()

    print("\nDiffusion model training complete!")


if __name__ == "__main__":
    main()
