"""Boundary marginal heads: start, end, and inside logits per query.

All scores are dot-product attention between projected boundary/token states
and projected query states, scaled by ``1/sqrt(d)``. Masks use
Masks use a shared finite sentinel so sums remain finite in mixed precision. No tensor is ``[L, L]``:
``inside_prefix`` is a cumulative sum of length ``L + 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from gliner2.models.boundary.constants import MASK_LOGIT


@dataclass
class BoundaryMarginals:
    start_logits: torch.Tensor    # [B, Q, L + 1]
    end_logits: torch.Tensor      # [B, Q, L + 1]
    inside_logits: torch.Tensor   # [B, Q, L]
    inside_prefix: torch.Tensor   # [B, Q, L + 1]
    inside_prefix_mean: torch.Tensor  # [B, Q, 1], restores centered intervals


def _masked_fill_min(logits: torch.Tensor, keep_mask: torch.BoolTensor) -> torch.Tensor:
    """Fill positions where ``keep_mask`` is False with a finite sentinel."""
    return logits.masked_fill(~keep_mask, MASK_LOGIT)


class BoundaryQueryHead(nn.Module):
    def __init__(self, hidden_size: int, boundary_dim: int, query_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.boundary_dim = boundary_dim
        q_dim = query_dim if query_dim is not None else hidden_size

        self.start_boundary_projection = nn.Linear(boundary_dim, boundary_dim)
        self.start_query_projection = nn.Linear(q_dim, boundary_dim)
        self.end_boundary_projection = nn.Linear(boundary_dim, boundary_dim)
        self.end_query_projection = nn.Linear(q_dim, boundary_dim)
        self.inside_text_projection = nn.Linear(hidden_size, boundary_dim)
        self.inside_query_projection = nn.Linear(q_dim, boundary_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        boundary_states: torch.Tensor,   # [B, L+1, d]
        boundary_mask: torch.BoolTensor,  # [B, L+1]
        text_states: torch.Tensor,        # [B, L, H]
        text_mask: torch.BoolTensor,      # [B, L]
        query_states: torch.Tensor,       # [B, Q, Hq]
        query_mask: torch.BoolTensor,     # [B, Q]
    ) -> BoundaryMarginals:
        scale = 1.0 / math.sqrt(self.boundary_dim)

        start_b = self.dropout(self.start_boundary_projection(boundary_states))  # [B,L+1,d]
        start_q = self.start_query_projection(query_states)                      # [B,Q,d]
        start_logits = torch.einsum("bld,bqd->bql", start_b, start_q) * scale     # [B,Q,L+1]

        end_b = self.dropout(self.end_boundary_projection(boundary_states))
        end_q = self.end_query_projection(query_states)
        end_logits = torch.einsum("bld,bqd->bql", end_b, end_q) * scale

        inside_t = self.dropout(self.inside_text_projection(text_states))         # [B,L,d]
        inside_q = self.inside_query_projection(query_states)                     # [B,Q,d]
        inside_logits = torch.einsum("bld,bqd->bql", inside_t, inside_q) * scale  # [B,Q,L]

        # Mask invalid boundaries/tokens and invalid queries.
        b_keep = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)            # [B,Q,L+1]
        t_keep = text_mask.unsqueeze(1) & query_mask.unsqueeze(-1)                # [B,Q,L]
        start_logits = _masked_fill_min(start_logits, b_keep)
        end_logits = _masked_fill_min(end_logits, b_keep)
        inside_logits = _masked_fill_min(inside_logits, t_keep)

        # Inside prefix: cumulative sum over tokens; zero-out masked positions so
        # the prefix difference over [i, j) equals the sum of real inside scores.
        inside_for_prefix = inside_logits.masked_fill(~t_keep, 0.0)
        # Center each query before an explicitly-fp32 cumulative sum. The mean
        # is carried separately and restored by interval scoring, preserving
        # exact values while preventing a large running offset.
        inside_for_prefix = inside_for_prefix.float()
        valid_count = t_keep.sum(-1, keepdim=True).clamp_min(1)
        inside_mean = (
            inside_for_prefix.sum(-1, keepdim=True) / valid_count
        ).detach()
        centered = (
            inside_for_prefix - inside_mean
        ) * t_keep.to(inside_for_prefix.dtype)
        zeros = torch.zeros(
            centered.shape[0], centered.shape[1], 1,
            dtype=torch.float32, device=inside_for_prefix.device,
        )
        inside_prefix = torch.cat([zeros, centered.cumsum(dim=-1)], dim=-1)

        return BoundaryMarginals(
            start_logits=start_logits,
            end_logits=end_logits,
            inside_logits=inside_logits,
            inside_prefix=inside_prefix,
            inside_prefix_mean=inside_mean,
        )
