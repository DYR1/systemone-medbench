"""Rotary embeddings indexed by token-boundary position."""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryBoundaryEmbedding(nn.Module):
    """Rotate endpoint vectors so dot products encode relative distance."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(f"rotary dim must be even, got {dim}")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, states: torch.Tensor, positions: torch.LongTensor
    ) -> torch.Tensor:
        angle = positions.unsqueeze(-1).float() * self.inv_freq
        cos, sin = torch.cos(angle), torch.sin(angle)
        even = states[..., 0::2].float()
        odd = states[..., 1::2].float()
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        )
        return rotated.flatten(-2).to(states.dtype)


__all__ = ["RotaryBoundaryEmbedding"]
