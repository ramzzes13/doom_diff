"""Step 6: Train the baseline (no memory) diffusion model for ablation comparison.

This script trains a MemGameNGen model with memory_enabled=False to compare
against the full memory-augmented model. All other hyperparameters are identical
to the memory-augmented training (03_train_diffusion.py).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import time
from memgamengen.data_collector import DoomTrajectoryDataset
from memgamengen.diffusion_model import MemGameNGen
from memgamengen.trainer import MemGameNGenTrainer


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Use GPU 3 (most free memory ~9.5GB)
    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (physical GPU {gpu_id})")

    # Check GPU memory
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
        total_mem = torch.cuda.mem_get_info(0)[1] / 1024**3
        print(f"GPU memory: {free_mem:.1f} GB free / {total_mem:.1f} GB total")

    # Load dataset
    data_dir = os.path.join(project_dir, "data", "trajectories")
    print(f"\nLoading dataset from {data_dir}...")

    full_dataset = DoomTrajectoryDataset(
        data_dir=data_dir,
        num_context_frames=4,
        resolution=(120, 160),
    )

    # Split into train/val (same split as memory model for fair comparison)
    total_len = len(full_dataset)
    val_len = max(1, int(total_len * 0.1))
    train_len = total_len - val_len
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_len}, Val: {val_len}")

    # Create baseline model (memory_enabled=False, everything else identical)
    print("\nCreating baseline MemGameNGen model (NO memory)...")
    model = MemGameNGen(
        pretrained_model="CompVis/stable-diffusion-v1-4",
        num_actions=8,
        num_context_frames=4,
        # Memory DISABLED for baseline
        memory_enabled=False,
        num_memory_tokens=16,  # unused but kept for consistent config
        memory_dim=768,
        memory_update_type="gru",
        # Action config (same)
        action_embed_dim=128,
        # LoRA config (same)
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        # State head (same, but uses frame features instead of memory)
        state_head_enabled=True,
        num_state_variables=2,  # HEALTH, AMMO2
        # Noise augmentation (same)
        noise_augmentation=True,
        noise_aug_max_level=0.7,
    )

    print(f"Trainable parameters: {model.count_trainable_params():,}")
    print(f"Memory module: {'ENABLED' if model.memory_enabled else 'DISABLED (baseline)'}")

    # Setup save directories
    save_dir = os.path.join(project_dir, "checkpoints", "baseline")
    log_dir = os.path.join(project_dir, "results", "logs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create trainer (identical hyperparameters to memory model)
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
        save_dir=save_dir,
        log_dir=log_dir,  # will save to training_log.json by default
        save_every=2000,
        eval_every=1000,
        log_every=50,
        mixed_precision=True,
        gradient_checkpointing=True,
        device=device,
        seed=42,
    )

    # Override the log save path to use baseline-specific filename
    original_save_log = trainer._save_training_log

    def save_baseline_log():
        """Save training metrics to baseline-specific JSON file."""
        log_path = os.path.join(log_dir, "baseline_training_log.json")
        log_data = {
            "train_losses": trainer.train_losses,
            "val_losses": trainer.val_losses,
            "config": {
                "lr": trainer.lr,
                "batch_size": trainer.batch_size,
                "gradient_accumulation_steps": trainer.gradient_accumulation_steps,
                "max_train_steps": trainer.max_train_steps,
                "diffusion_loss_weight": trainer.diffusion_loss_weight,
                "state_loss_weight": trainer.state_loss_weight,
                "memory_enabled": False,
                "model_type": "baseline_no_memory",
            }
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"Saved baseline training log: {log_path}")

    trainer._save_training_log = save_baseline_log

    # Train
    print("\n" + "=" * 60)
    print("BASELINE TRAINING (NO MEMORY)")
    print("=" * 60)
    start_time = time.time()

    trainer.train()

    total_time = time.time() - start_time
    print(f"\nBaseline training complete!")
    print(f"Total training time: {total_time / 60:.1f} minutes")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"Training log saved to: {os.path.join(log_dir, 'baseline_training_log.json')}")


if __name__ == "__main__":
    main()
