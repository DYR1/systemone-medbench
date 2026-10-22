"""Boundary state encoding.

For a text of ``n`` tokens there are ``n + 1`` boundaries ``0..n``. Boundary
``i`` sits between token ``i - 1`` (its left) and token ``i`` (its right). The
first boundary uses a learned BOS left-state; the final boundary uses a learned
EOS right-state. Padding boundaries beyond a sample's length are masked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BoundaryEncoding:
    states: torch.Tensor      # [B, L + 1, d]
    mask: torch.BoolTensor    # [B, L + 1]


def build_boundary_mask(text_lengths: torch.LongTensor, max_text_length: int) -> torch.BoolTensor:
    """Boundary validity mask ``[B, L + 1]``: boundary ``i`` valid iff ``i <= n``."""
    device = text_lengths.device
    idx = torch.arange(max_text_length + 1, device=device).unsqueeze(0)  # [1, L+1]
    return idx <= text_lengths.unsqueeze(1)


def shift_left_with_bos(text_states: torch.Tensor, bos_state: torch.Tensor) -> torch.Tensor:
    """Left states for each boundary: ``left(i) = token i-1``; ``left(0) = BOS``.

    Args:
        text_states: [B, L, H]
        bos_state:   [H]
    Returns:
        [B, L + 1, H]
    """
    b, l, h = text_states.shape
    out = torch.empty(b, l + 1, h, dtype=text_states.dtype, device=text_states.device)
    out[:, 0] = bos_state.to(text_states.dtype)
    out[:, 1:] = text_states
    return out


def shift_right_with_eos(
    text_states: torch.Tensor,
    text_lengths: torch.LongTensor,
    eos_state: torch.Tensor,
) -> torch.Tensor:
    """Right states for each boundary: ``right(i) = token i``; ``right(n) = EOS``.

    The EOS state is placed at each sample's own final boundary ``n_b``.

    Args:
        text_states:  [B, L, H]
        text_lengths: [B]
        eos_state:    [H]
    Returns:
        [B, L + 1, H]
    """
    b, l, h = text_states.shape
    out = torch.empty(b, l + 1, h, dtype=text_states.dtype, device=text_states.device)
    out[:, :l] = text_states
    out[:, l] = eos_state.to(text_states.dtype)
    # Place EOS at each sample's final boundary index n_b.
    eos = eos_state.to(text_states.dtype).view(1, h).expand(b, h)
    out[torch.arange(b, device=text_states.device), text_lengths.clamp(max=l)] = eos
    return out


class ResidualSwiGLU(nn.Module):
    """Compact pre-norm residual SwiGLU feed-forward block."""

    def __init__(self, dim: int, multiplier: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(1, int(dim * multiplier))
        self.norm = nn.LayerNorm(dim)
        self.input_projection = nn.Linear(dim, 2 * hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        value, gate = self.input_projection(self.norm(states)).chunk(2, dim=-1)
        update = value * F.silu(gate)
        update = self.dropout(update)
        update = self.output_projection(update)
        return states + self.dropout(update)


class BoundaryAttentionBlock(nn.Module):
    """Pre-norm self-attention over valid boundary positions."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.norm = nn.LayerNorm(dim)
        self.qkv_projection = nn.Linear(dim, 3 * dim)
        self.output_projection = nn.Linear(dim, dim)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, states: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        b, n, d = states.shape
        qkv = self.qkv_projection(self.norm(states)).view(
            b, n, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        allowed = mask.view(b, 1, 1, n).expand(b, 1, n, n)
        if self.window > 0:
            positions = torch.arange(n, device=states.device)
            local = (
                positions.view(n, 1) - positions.view(1, n)
            ).abs() <= self.window
            allowed = allowed & local.view(1, 1, n, n)
        # Padding query rows still need one legal key to avoid NaNs.
        diagonal = torch.eye(n, dtype=torch.bool, device=states.device)
        allowed = allowed | diagonal.view(1, 1, n, n)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(b, n, d)
        update = self.dropout(self.output_projection(attended))
        return (states + update) * mask.unsqueeze(-1).to(states.dtype)


class BoundaryEncoder(nn.Module):
    """Projects and refines left/right token states into per-boundary states."""

    def __init__(
        self,
        hidden_size: int,
        boundary_dim: int,
        dropout: float = 0.1,
        refinement_layers: int = 1,
        ffn_multiplier: float = 2.0,
        attention_layers: int = 0,
        attention_heads: int = 4,
        attention_window: int = 0,
    ):
        super().__init__()
        if refinement_layers < 0:
            raise ValueError(f"refinement_layers must be >= 0, got {refinement_layers}")
        if ffn_multiplier <= 0:
            raise ValueError(f"ffn_multiplier must be > 0, got {ffn_multiplier}")
        self.hidden_size = hidden_size
        self.boundary_dim = boundary_dim
        self.left_projection = nn.Linear(hidden_size, boundary_dim)
        self.right_projection = nn.Linear(hidden_size, boundary_dim)
        self.output_projection = nn.Linear(2 * boundary_dim, boundary_dim)
        self.layer_norm = nn.LayerNorm(boundary_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_blocks = nn.ModuleList(
            BoundaryAttentionBlock(
                boundary_dim, attention_heads, attention_window, dropout
            )
            for _ in range(attention_layers)
        )
        self.refinement_blocks = nn.ModuleList(
            ResidualSwiGLU(boundary_dim, ffn_multiplier, dropout)
            for _ in range(refinement_layers)
        )
        self.bos_state = nn.Parameter(torch.zeros(hidden_size))
        self.eos_state = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.bos_state, std=0.02)
        nn.init.normal_(self.eos_state, std=0.02)

    def forward(self, text_states: torch.Tensor, text_mask: torch.BoolTensor) -> BoundaryEncoding:
        b, l, h = text_states.shape
        text_lengths = text_mask.sum(dim=1).long()  # [B]

        left = shift_left_with_bos(text_states, self.bos_state)     # [B, L+1, H]
        right = shift_right_with_eos(text_states, text_lengths, self.eos_state)

        left_p = self.left_projection(left)
        right_p = self.right_projection(right)
        states = self.output_projection(torch.cat([left_p, right_p], dim=-1))
        states = self.layer_norm(states)
        states = self.dropout(states)
        mask = build_boundary_mask(text_lengths, l)  # [B, L+1]
        for block in self.attention_blocks:
            states = block(states, mask)
        for block in self.refinement_blocks:
            states = block(states)

        # Zero out padding boundary states so they cannot leak into downstream
        # gathers/scores through numerical noise.
        states = states * mask.unsqueeze(-1).to(states.dtype)
        return BoundaryEncoding(states=states, mask=mask)
