"""Memory-Augmented Action-Conditioned Diffusion Model for DOOM.

This implements the core MemGameNGen model: a latent diffusion model that generates
next frames conditioned on past frames, actions, and persistent memory tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig, get_peft_model
import math


class ActionEmbedding(nn.Module):
    """Embeds discrete actions into continuous vectors for cross-attention conditioning."""

    def __init__(self, num_actions: int = 8, embed_dim: int = 128, cross_attn_dim: int = 768):
        super().__init__()
        self.embed = nn.Embedding(num_actions, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, cross_attn_dim),
            nn.SiLU(),
            nn.Linear(cross_attn_dim, cross_attn_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, N) action indices for N context frames
        Returns:
            (B, N, cross_attn_dim) action embeddings for cross-attention
        """
        x = self.embed(actions)  # (B, N, embed_dim)
        return self.proj(x)  # (B, N, cross_attn_dim)


class NoiseAugmentation(nn.Module):
    """Conditioning noise augmentation from GameNGen.

    During training, corrupt context frames with known noise level.
    The noise level is embedded and provided to the model so it can account for corruption.
    """

    def __init__(self, max_noise_level: float = 0.7, embed_dim: int = 768):
        super().__init__()
        self.max_noise_level = max_noise_level
        # Sinusoidal embedding for noise level
        self.noise_level_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def add_noise(self, latents: torch.Tensor, noise_level: Optional[float] = None):
        """Add noise to context latents during training."""
        if noise_level is None:
            noise_level = torch.rand(1).item() * self.max_noise_level

        noise = torch.randn_like(latents) * noise_level
        noisy_latents = latents + noise
        return noisy_latents, noise_level

    def get_noise_embedding(self, noise_level: float, batch_size: int, device: torch.device):
        """Get noise level embedding."""
        level_tensor = torch.tensor([[noise_level]], device=device).expand(batch_size, -1)
        return self.noise_level_embed(level_tensor)  # (B, embed_dim)


class StateHead(nn.Module):
    """Auxiliary head that predicts game state variables from memory/features.

    This provides state-consistency supervision: the model must maintain
    correct game state in its internal representations.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_variables: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_variables),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) memory or pooled features
        Returns:
            (B, num_variables) predicted game state variables
        """
        return self.head(features)


class LatentFrameEncoder(nn.Module):
    """Encodes latent frames into feature vectors for memory update."""

    def __init__(self, latent_channels: int = 4, latent_h: int = 15, latent_w: int = 20,
                 output_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(latent_channels, 32, 3, stride=2, padding=1),  # -> (32, 8, 10)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> (64, 4, 5)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> (64, 1, 1)
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, C, H, W) latent frame
        Returns:
            (B, output_dim) feature vector
        """
        x = self.conv(latent)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class MemGameNGen(nn.Module):
    """Memory-Augmented Game Neural Generator.

    Architecture:
    - VAE encoder/decoder from Stable Diffusion (frozen, with optional decoder fine-tuning)
    - UNet2D with LoRA for action-conditioned next-frame prediction
    - Memory module maintaining persistent state across frames
    - Action embedding via cross-attention
    - State head for auxiliary game-state prediction
    - Noise augmentation for context frames (GameNGen technique)
    """

    def __init__(
        self,
        pretrained_model: str = "CompVis/stable-diffusion-v1-4",
        num_actions: int = 8,
        num_context_frames: int = 4,
        # Memory config
        memory_enabled: bool = True,
        num_memory_tokens: int = 16,
        memory_dim: int = 768,
        memory_update_type: str = "gru",
        # Action config
        action_embed_dim: int = 128,
        # LoRA config
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        # State head
        state_head_enabled: bool = True,
        num_state_variables: int = 4,
        # Noise augmentation
        noise_augmentation: bool = True,
        noise_aug_max_level: float = 0.7,
        # Misc
        latent_channels: int = 4,
    ):
        super().__init__()

        self.num_context_frames = num_context_frames
        self.memory_enabled = memory_enabled
        self.noise_augmentation_enabled = noise_augmentation
        self.state_head_enabled = state_head_enabled
        self.latent_channels = latent_channels

        # Load VAE (frozen)
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model, subfolder="vae"
        )
        self.vae.requires_grad_(False)
        self.vae_scale_factor = 0.18215

        # Load UNet
        print("Loading UNet...")
        # We need to modify UNet to accept concatenated context latents
        # The original SD UNet has in_channels=4; we need 4 * (num_context_frames + 1) = 20 for 4 context frames
        # However, modifying in_channels directly is complex. Instead, we'll use a projection layer.
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder="unet"
        )

        # Input projection: concatenated context latents -> 4 channels
        total_input_channels = latent_channels * (num_context_frames + 1)  # context + noisy target
        self.input_proj = nn.Conv2d(
            total_input_channels, 4, kernel_size=1, bias=True
        )
        # Initialize to roughly pass through the noisy target
        nn.init.zeros_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        with torch.no_grad():
            # Initialize so that the last 4 channels (noisy target) pass through
            self.input_proj.weight[:, -latent_channels:, :, :] = torch.eye(4).reshape(4, 4, 1, 1)

        # Apply LoRA to UNet
        if use_lora:
            print("Applying LoRA to UNet...")
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                lora_dropout=0.05,
            )
            self.unet = get_peft_model(self.unet, lora_config)
            self.unet.print_trainable_parameters()
        else:
            # Freeze most of UNet, only train attention layers
            for name, param in self.unet.named_parameters():
                if "attn" not in name:
                    param.requires_grad = False

        # Action embedding
        self.action_embed = ActionEmbedding(
            num_actions=num_actions,
            embed_dim=action_embed_dim,
            cross_attn_dim=self.unet.config.cross_attention_dim,
        )

        # Memory module
        if memory_enabled:
            from .memory_module import MemoryModule
            self.memory = MemoryModule(
                memory_dim=memory_dim,
                latent_dim=256,
                action_dim=action_embed_dim,
                num_memory_tokens=num_memory_tokens,
                update_type=memory_update_type,
            )
            # Project action for memory update
            self.memory_action_embed = nn.Embedding(num_actions, action_embed_dim)
        else:
            self.memory = None

        # Latent frame encoder (for memory update)
        self.frame_encoder = LatentFrameEncoder(
            latent_channels=latent_channels,
            output_dim=256,
        )

        # Noise augmentation
        if noise_augmentation:
            self.noise_aug = NoiseAugmentation(
                max_noise_level=noise_aug_max_level,
                embed_dim=self.unet.config.cross_attention_dim,
            )
        else:
            self.noise_aug = None

        # State head
        if state_head_enabled:
            self.state_head = StateHead(
                input_dim=memory_dim if memory_enabled else 256,
                hidden_dim=256,
                num_variables=num_state_variables,
            )
        else:
            self.state_head = None

        # Noise scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            prediction_type="epsilon",
        )

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames to VAE latents.

        Args:
            frames: (B, C, H, W) or (B, N, C, H, W) normalized to [-1, 1]
        Returns:
            latents: same shape with C=4 and reduced spatial dims
        """
        if frames.dim() == 5:
            B, N, C, H, W = frames.shape
            frames_flat = frames.reshape(B * N, C, H, W)
            latents_flat = self.vae.encode(frames_flat).latent_dist.sample()
            latents_flat = latents_flat * self.vae_scale_factor
            _, c, h, w = latents_flat.shape
            return latents_flat.reshape(B, N, c, h, w)
        else:
            latents = self.vae.encode(frames).latent_dist.sample()
            return latents * self.vae_scale_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to pixel frames.

        Args:
            latents: (B, 4, H, W)
        Returns:
            frames: (B, 3, H_orig, W_orig) in [-1, 1]
        """
        latents = latents / self.vae_scale_factor
        frames = self.vae.decode(latents).sample
        return frames

    def build_cross_attention_context(
        self,
        action_embeds: torch.Tensor,  # (B, N, D)
        memory_tokens: Optional[torch.Tensor] = None,  # (B, K, D)
        noise_level_embed: Optional[torch.Tensor] = None,  # (B, D)
    ) -> torch.Tensor:
        """Build the cross-attention context for the UNet.

        Concatenates action embeddings, memory tokens, and noise level embedding.
        """
        parts = [action_embeds]

        if memory_tokens is not None:
            parts.append(memory_tokens)

        if noise_level_embed is not None:
            parts.append(noise_level_embed.unsqueeze(1))  # (B, 1, D)

        return torch.cat(parts, dim=1)  # (B, N+K+1, D)

    def forward(
        self,
        context_frames: torch.Tensor,  # (B, N, 3, H, W) in [-1, 1]
        target_frame: torch.Tensor,  # (B, 3, H, W) in [-1, 1]
        context_actions: torch.Tensor,  # (B, N) action indices
        memory_state: Optional[torch.Tensor] = None,  # (B, K, D)
        game_variables: Optional[torch.Tensor] = None,  # (B, num_vars)
        noise_level: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Returns dict with 'diffusion_loss', 'state_loss', 'predicted_noise', etc.
        """
        B = context_frames.shape[0]
        device = context_frames.device

        # Encode frames to latents
        context_latents = self.encode_frames(context_frames)  # (B, N, 4, h, w)
        target_latents = self.encode_frames(target_frame)  # (B, 4, h, w)

        # Apply noise augmentation to context latents
        if self.noise_augmentation_enabled and self.training:
            B_ctx, N, C, h, w = context_latents.shape
            context_flat = context_latents.reshape(B_ctx * N, C, h, w)
            context_flat, actual_noise_level = self.noise_aug.add_noise(context_flat, noise_level)
            context_latents = context_flat.reshape(B_ctx, N, C, h, w)
        else:
            actual_noise_level = 0.0

        # Add diffusion noise to target
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=device).long()
        noisy_target = self.scheduler.add_noise(target_latents, noise, timesteps)

        # Concatenate context latents with noisy target along channel dim
        # context_latents: (B, N, 4, h, w) -> reshape to (B, N*4, h, w)
        B_ctx, N, C, h, w = context_latents.shape
        context_flat = context_latents.reshape(B, N * C, h, w)
        unet_input = torch.cat([context_flat, noisy_target], dim=1)  # (B, (N+1)*4, h, w)

        # Project to 4 channels
        unet_input = self.input_proj(unet_input)  # (B, 4, h, w)

        # Build cross-attention context
        action_embeds = self.action_embed(context_actions)  # (B, N, D)

        noise_level_embed = None
        if self.noise_aug is not None:
            noise_level_embed = self.noise_aug.get_noise_embedding(
                actual_noise_level, B, device
            )

        encoder_hidden_states = self.build_cross_attention_context(
            action_embeds, memory_state, noise_level_embed
        )

        # UNet forward
        noise_pred = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        # Diffusion loss
        diffusion_loss = F.mse_loss(noise_pred, noise)

        results = {
            "diffusion_loss": diffusion_loss,
            "noise_pred": noise_pred,
            "target_latents": target_latents,
        }

        # State prediction loss
        if self.state_head_enabled and game_variables is not None:
            if memory_state is not None:
                # Pool memory tokens
                state_features = memory_state.mean(dim=1)  # (B, D)
            else:
                # Use frame features
                state_features = self.frame_encoder(target_latents)  # (B, 256)

            state_pred = self.state_head(state_features)  # (B, num_vars)
            state_loss = F.mse_loss(state_pred, game_variables)
            results["state_loss"] = state_loss
            results["state_pred"] = state_pred

        return results

    @torch.no_grad()
    def generate_next_frame(
        self,
        context_frames: torch.Tensor,  # (B, N, 3, H, W) in [-1, 1]
        context_actions: torch.Tensor,  # (B, N) action indices
        memory_state: Optional[torch.Tensor] = None,  # (B, K, D)
        num_inference_steps: int = 4,
        last_action: Optional[torch.Tensor] = None,  # (B,) for memory update
    ) -> Dict[str, torch.Tensor]:
        """Generate the next frame using DDIM sampling.

        Returns dict with 'frame', 'latent', 'memory_state', 'state_pred'.
        """
        B = context_frames.shape[0]
        device = context_frames.device

        # Encode context frames
        context_latents = self.encode_frames(context_frames)  # (B, N, 4, h, w)
        B_ctx, N, C, h, w = context_latents.shape

        # Build cross-attention context
        action_embeds = self.action_embed(context_actions)  # (B, N, D)
        noise_level_embed = None
        if self.noise_aug is not None:
            noise_level_embed = self.noise_aug.get_noise_embedding(0.0, B, device)

        encoder_hidden_states = self.build_cross_attention_context(
            action_embeds, memory_state, noise_level_embed
        )

        # Prepare context for UNet
        context_flat = context_latents.reshape(B, N * C, h, w)

        # DDIM sampling
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        latent = torch.randn(B, 4, h, w, device=device)

        for t in self.scheduler.timesteps:
            # Concatenate context with current noisy latent
            unet_input = torch.cat([context_flat, latent], dim=1)
            unet_input = self.input_proj(unet_input)

            # Predict noise
            noise_pred = self.unet(
                unet_input,
                t.unsqueeze(0).expand(B),
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # DDIM step
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample

        # Decode
        frame = self.decode_latents(latent)

        results = {
            "frame": frame,
            "latent": latent,
        }

        # Update memory
        if self.memory is not None and memory_state is not None and last_action is not None:
            frame_features = self.frame_encoder(latent)  # (B, 256)
            action_embed_for_mem = self.memory_action_embed(last_action)  # (B, action_dim)
            new_memory = self.memory.update(memory_state, frame_features, action_embed_for_mem)
            results["memory_state"] = new_memory
        else:
            results["memory_state"] = memory_state

        # State prediction
        if self.state_head is not None and memory_state is not None:
            state_features = results["memory_state"].mean(dim=1) if results["memory_state"] is not None else self.frame_encoder(latent)
            results["state_pred"] = self.state_head(state_features)

        return results

    def get_trainable_params(self):
        """Get all trainable parameters."""
        params = []
        # Input projection
        params.extend(self.input_proj.parameters())
        # UNet (LoRA params)
        for p in self.unet.parameters():
            if p.requires_grad:
                params.append(p)
        # Action embedding
        params.extend(self.action_embed.parameters())
        # Memory
        if self.memory is not None:
            params.extend(self.memory.parameters())
            params.extend(self.memory_action_embed.parameters())
        # Frame encoder
        params.extend(self.frame_encoder.parameters())
        # Noise augmentation
        if self.noise_aug is not None:
            params.extend(self.noise_aug.parameters())
        # State head
        if self.state_head is not None:
            params.extend(self.state_head.parameters())
        return params

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())
