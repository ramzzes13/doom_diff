"""Memory modules for MemGameNGen - maintains persistent state across frames."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GRUMemoryUpdate(nn.Module):
    """GRU-based memory update: m_t = GRU(m_{t-1}, [z_t, a_t])."""

    def __init__(
        self,
        memory_dim: int = 768,
        latent_dim: int = 256,
        action_dim: int = 128,
        num_memory_tokens: int = 16,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_memory_tokens = num_memory_tokens

        # Project frame latent to input dim
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, memory_dim),
            nn.SiLU(),
        )

        # Project action to input dim
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, memory_dim),
            nn.SiLU(),
        )

        # Combine frame + action
        self.input_proj = nn.Linear(memory_dim * 2, memory_dim)

        # GRU cell for each memory token
        self.gru = nn.GRUCell(memory_dim, memory_dim)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(memory_dim)

    def forward(
        self,
        memory: torch.Tensor,  # (B, K, D) - previous memory tokens
        frame_features: torch.Tensor,  # (B, latent_dim) - current frame features
        action_embed: torch.Tensor,  # (B, action_dim) - current action embedding
    ) -> torch.Tensor:
        """Update memory tokens given new frame and action."""
        B, K, D = memory.shape

        # Project inputs
        frame_proj = self.latent_proj(frame_features)  # (B, D)
        action_proj = self.action_proj(action_embed)  # (B, D)

        # Combine into input
        combined = self.input_proj(torch.cat([frame_proj, action_proj], dim=-1))  # (B, D)

        # Update each memory token with GRU
        # Broadcast the input across all K tokens
        combined_expanded = combined.unsqueeze(1).expand(-1, K, -1)  # (B, K, D)

        # Reshape for GRU
        memory_flat = memory.reshape(B * K, D)
        input_flat = combined_expanded.reshape(B * K, D)

        # GRU update
        new_memory_flat = self.gru(input_flat, memory_flat)

        # Reshape back
        new_memory = new_memory_flat.reshape(B, K, D)
        new_memory = self.layer_norm(new_memory)

        return new_memory


class CrossAttentionMemoryUpdate(nn.Module):
    """Cross-attention based memory update."""

    def __init__(
        self,
        memory_dim: int = 768,
        latent_dim: int = 256,
        action_dim: int = 128,
        num_memory_tokens: int = 16,
        num_heads: int = 8,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_memory_tokens = num_memory_tokens

        # Project frame features
        self.frame_proj = nn.Linear(latent_dim, memory_dim)
        self.action_proj = nn.Linear(action_dim, memory_dim)

        # Cross-attention: memory queries attend to frame+action keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(memory_dim, memory_dim * 4),
            nn.GELU(),
            nn.Linear(memory_dim * 4, memory_dim),
        )

        self.norm1 = nn.LayerNorm(memory_dim)
        self.norm2 = nn.LayerNorm(memory_dim)

    def forward(
        self,
        memory: torch.Tensor,  # (B, K, D)
        frame_features: torch.Tensor,  # (B, latent_dim)
        action_embed: torch.Tensor,  # (B, action_dim)
    ) -> torch.Tensor:
        B, K, D = memory.shape

        # Create key-value pairs from frame + action
        frame_kv = self.frame_proj(frame_features).unsqueeze(1)  # (B, 1, D)
        action_kv = self.action_proj(action_embed).unsqueeze(1)  # (B, 1, D)
        kv = torch.cat([frame_kv, action_kv], dim=1)  # (B, 2, D)

        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=self.norm1(memory),
            key=kv,
            value=kv,
        )
        memory = memory + attn_out

        # Feed-forward
        memory = memory + self.ffn(self.norm2(memory))

        return memory


class MemoryModule(nn.Module):
    """Top-level memory module combining initialization, update, and readout."""

    def __init__(
        self,
        memory_dim: int = 768,
        latent_dim: int = 256,
        action_dim: int = 128,
        num_memory_tokens: int = 16,
        update_type: str = "gru",
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_memory_tokens = num_memory_tokens

        # Learnable initial memory
        self.initial_memory = nn.Parameter(
            torch.randn(1, num_memory_tokens, memory_dim) * 0.02
        )

        # Memory update module
        if update_type == "gru":
            self.update_module = GRUMemoryUpdate(
                memory_dim=memory_dim,
                latent_dim=latent_dim,
                action_dim=action_dim,
                num_memory_tokens=num_memory_tokens,
            )
        elif update_type == "cross_attention":
            self.update_module = CrossAttentionMemoryUpdate(
                memory_dim=memory_dim,
                latent_dim=latent_dim,
                action_dim=action_dim,
                num_memory_tokens=num_memory_tokens,
            )
        else:
            raise ValueError(f"Unknown update type: {update_type}")

    def get_initial_memory(self, batch_size: int) -> torch.Tensor:
        """Get initial memory tokens for a batch."""
        return self.initial_memory.expand(batch_size, -1, -1)

    def update(
        self,
        memory: torch.Tensor,
        frame_features: torch.Tensor,
        action_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Update memory with new frame and action."""
        return self.update_module(memory, frame_features, action_embed)
