"""Training pipeline for MemGameNGen with state-consistency supervision."""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
import json


class MemGameNGenTrainer:
    """Trainer for the memory-augmented diffusion world model."""

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        # Training hyperparams
        lr: float = 1e-4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_train_steps: int = 20000,
        warmup_steps: int = 500,
        # Loss weights
        diffusion_loss_weight: float = 1.0,
        state_loss_weight: float = 0.1,
        # Misc
        save_dir: str = "checkpoints/diffusion",
        log_dir: str = "results/logs",
        save_every: int = 2000,
        eval_every: int = 1000,
        log_every: int = 50,
        mixed_precision: bool = True,
        gradient_checkpointing: bool = True,
        device: torch.device = torch.device("cuda:0"),
        seed: int = 42,
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_train_steps = max_train_steps
        self.warmup_steps = warmup_steps
        self.diffusion_loss_weight = diffusion_loss_weight
        self.state_loss_weight = state_loss_weight
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.save_every = save_every
        self.eval_every = eval_every
        self.log_every = log_every
        self.mixed_precision = mixed_precision
        self.seed = seed

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        torch.manual_seed(seed)

        # Move model to device
        self.model = self.model.to(device)

        # Enable gradient checkpointing
        if gradient_checkpointing:
            if hasattr(self.model.unet, 'enable_gradient_checkpointing'):
                self.model.unet.enable_gradient_checkpointing()

        # Optimizer - only trainable params
        self.optimizer = torch.optim.AdamW(
            self.model.get_trainable_params(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_train_steps,
            eta_min=lr * 0.1,
        )

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None

        # Training state
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []

    def _get_warmup_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * step / self.warmup_steps
        return self.lr

    def _apply_warmup(self, step: int):
        if step < self.warmup_steps:
            lr = self._get_warmup_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        context_frames = batch["context_frames"].to(self.device)  # (B, N, 3, H, W)
        target_frame = batch["target_frame"].to(self.device)  # (B, 3, H, W)
        context_actions = batch["context_actions"].to(self.device)  # (B, N)
        game_variables = batch["game_variables"].to(self.device) if "game_variables" in batch else None

        # Initialize memory for this batch
        B = context_frames.shape[0]
        memory_state = None
        if self.model.memory is not None:
            memory_state = self.model.memory.get_initial_memory(B).to(self.device)

        # Forward pass
        if self.mixed_precision:
            with torch.amp.autocast('cuda'):
                outputs = self.model(
                    context_frames=context_frames,
                    target_frame=target_frame,
                    context_actions=context_actions,
                    memory_state=memory_state,
                    game_variables=game_variables,
                )
        else:
            outputs = self.model(
                context_frames=context_frames,
                target_frame=target_frame,
                context_actions=context_actions,
                memory_state=memory_state,
                game_variables=game_variables,
            )

        # Compute total loss
        total_loss = self.diffusion_loss_weight * outputs["diffusion_loss"]

        if "state_loss" in outputs:
            total_loss = total_loss + self.state_loss_weight * outputs["state_loss"]

        # Scale loss for gradient accumulation
        total_loss = total_loss / self.gradient_accumulation_steps

        # Backward
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        loss_dict = {
            "total_loss": total_loss.item() * self.gradient_accumulation_steps,
            "diffusion_loss": outputs["diffusion_loss"].item(),
        }
        if "state_loss" in outputs:
            loss_dict["state_loss"] = outputs["state_loss"].item()

        return loss_dict

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting MemGameNGen Training")
        print(f"  Trainable params: {self.model.count_trainable_params():,}")
        print(f"  Max steps: {self.max_train_steps}")
        print(f"  Batch size: {self.batch_size} x {self.gradient_accumulation_steps} (acc)")
        print(f"  Learning rate: {self.lr}")
        print(f"  Mixed precision: {self.mixed_precision}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        self.model.train()
        data_iter = iter(self.train_loader)
        accumulated_losses = {}
        start_time = time.time()

        while self.global_step < self.max_train_steps:
            self.optimizer.zero_grad()

            for acc_step in range(self.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                loss_dict = self.train_step(batch)

                for k, v in loss_dict.items():
                    if k not in accumulated_losses:
                        accumulated_losses[k] = 0
                    accumulated_losses[k] += v / self.gradient_accumulation_steps

            # Optimizer step
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_params(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_params(), 1.0)
                self.optimizer.step()

            self.global_step += 1

            # Learning rate
            self._apply_warmup(self.global_step)
            if self.global_step >= self.warmup_steps:
                self.lr_scheduler.step()

            # Logging
            if self.global_step % self.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed
                lr_current = self.optimizer.param_groups[0]['lr']

                log_msg = (
                    f"Step {self.global_step}/{self.max_train_steps} | "
                    f"Loss: {accumulated_losses.get('total_loss', 0):.4f} | "
                    f"Diff: {accumulated_losses.get('diffusion_loss', 0):.4f} | "
                )
                if "state_loss" in accumulated_losses:
                    log_msg += f"State: {accumulated_losses['state_loss']:.4f} | "
                log_msg += f"LR: {lr_current:.2e} | Steps/s: {steps_per_sec:.2f}"
                print(log_msg)

                self.train_losses.append({
                    "step": self.global_step,
                    **accumulated_losses,
                })

            accumulated_losses = {}

            # Save checkpoint
            if self.global_step % self.save_every == 0:
                self.save_checkpoint()

            # Validation
            if self.val_loader is not None and self.global_step % self.eval_every == 0:
                val_loss = self.validate()
                print(f"  Validation loss: {val_loss:.4f}")
                self.val_losses.append({"step": self.global_step, "val_loss": val_loss})
                self.model.train()

        # Final save
        self.save_checkpoint(tag="final")
        self._save_training_log()
        print(f"\nTraining complete! {self.global_step} steps in {time.time() - start_time:.1f}s")

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            context_frames = batch["context_frames"].to(self.device)
            target_frame = batch["target_frame"].to(self.device)
            context_actions = batch["context_actions"].to(self.device)
            game_variables = batch["game_variables"].to(self.device) if "game_variables" in batch else None

            B = context_frames.shape[0]
            memory_state = None
            if self.model.memory is not None:
                memory_state = self.model.memory.get_initial_memory(B).to(self.device)

            with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                outputs = self.model(
                    context_frames=context_frames,
                    target_frame=target_frame,
                    context_actions=context_actions,
                    memory_state=memory_state,
                    game_variables=game_variables,
                )

            loss = outputs["diffusion_loss"]
            if "state_loss" in outputs:
                loss = loss + self.state_loss_weight * outputs["state_loss"]

            total_loss += loss.item()
            num_batches += 1

            if num_batches >= 50:
                break

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, tag: Optional[str] = None):
        """Save model checkpoint."""
        if tag is None:
            tag = f"step_{self.global_step}"

        path = os.path.join(self.save_dir, f"memgamengen_{tag}.pt")

        # Save only trainable parameters + extra modules
        state_dict = {}

        # Input projection
        state_dict["input_proj"] = self.model.input_proj.state_dict()

        # Action embedding
        state_dict["action_embed"] = self.model.action_embed.state_dict()

        # Frame encoder
        state_dict["frame_encoder"] = self.model.frame_encoder.state_dict()

        # Memory
        if self.model.memory is not None:
            state_dict["memory"] = self.model.memory.state_dict()
            state_dict["memory_action_embed"] = self.model.memory_action_embed.state_dict()

        # State head
        if self.model.state_head is not None:
            state_dict["state_head"] = self.model.state_head.state_dict()

        # Noise augmentation
        if self.model.noise_aug is not None:
            state_dict["noise_aug"] = self.model.noise_aug.state_dict()

        # UNet LoRA weights
        unet_trainable = {
            k: v for k, v in self.model.unet.state_dict().items()
            if "lora" in k.lower()
        }
        state_dict["unet_lora"] = unet_trainable

        # Optimizer state
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["global_step"] = self.global_step

        torch.save(state_dict, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        state_dict = torch.load(path, map_location=self.device, weights_only=False)

        self.model.input_proj.load_state_dict(state_dict["input_proj"])
        self.model.action_embed.load_state_dict(state_dict["action_embed"])
        self.model.frame_encoder.load_state_dict(state_dict["frame_encoder"])

        if "memory" in state_dict and self.model.memory is not None:
            self.model.memory.load_state_dict(state_dict["memory"])
        if "memory_action_embed" in state_dict and hasattr(self.model, "memory_action_embed"):
            self.model.memory_action_embed.load_state_dict(state_dict["memory_action_embed"])
        if "state_head" in state_dict and self.model.state_head is not None:
            self.model.state_head.load_state_dict(state_dict["state_head"])
        if "noise_aug" in state_dict and self.model.noise_aug is not None:
            self.model.noise_aug.load_state_dict(state_dict["noise_aug"])

        # Load UNet LoRA weights
        if "unet_lora" in state_dict:
            current = self.model.unet.state_dict()
            current.update(state_dict["unet_lora"])
            self.model.unet.load_state_dict(current, strict=False)

        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "global_step" in state_dict:
            self.global_step = state_dict["global_step"]

        print(f"Loaded checkpoint from {path}, step {self.global_step}")

    def _save_training_log(self):
        """Save training metrics to JSON."""
        log_path = os.path.join(self.log_dir, "training_log.json")
        log_data = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": {
                "lr": self.lr,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_train_steps": self.max_train_steps,
                "diffusion_loss_weight": self.diffusion_loss_weight,
                "state_loss_weight": self.state_loss_weight,
            }
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"Saved training log: {log_path}")
