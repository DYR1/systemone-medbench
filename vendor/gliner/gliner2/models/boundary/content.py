"""Prefix-sum span-content pooling for sparse boundary candidates."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from gliner2.models.boundary.indexing import gather_rows, gather_states


def gather_prefix(prefix: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """Gather prefixes for per-query or document-pooled candidate indices."""
    if indices.dim() == 2:
        return gather_rows(prefix, indices)
    return gather_states(prefix, indices)


class SpanContentPooler(nn.Module):
    """Mean and optional smooth-maximum span pooling from token prefixes."""

    def __init__(
        self,
        hidden_size: int,
        content_dim: int,
        dropout: float = 0.1,
        use_soft_max_pool: bool = False,
    ) -> None:
        super().__init__()
        self.content_dim = content_dim
        self.use_soft_max_pool = use_soft_max_pool
        self.output_dim = content_dim * (2 if use_soft_max_pool else 1)
        self.value_projection = nn.Linear(hidden_size, content_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def build_prefix(
        self,
        text_states: torch.Tensor,
        text_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        values = self.value_projection(text_states)
        values = values * text_mask.unsqueeze(-1).to(values.dtype)
        values32 = values.float()
        zeros = values32.new_zeros(values32.shape[0], 1, self.content_dim)
        mean_prefix = torch.cat((zeros, values32.cumsum(1)), dim=1)

        lse_prefix = None
        if self.use_soft_max_pool:
            floor = torch.finfo(torch.float32).min / 4.0
            masked = values32.masked_fill(~text_mask.unsqueeze(-1), floor)
            lse_prefix = torch.cat(
                (
                    zeros.new_full((zeros.shape[0], 1, self.content_dim), floor),
                    torch.logcumsumexp(masked, dim=1),
                ),
                dim=1,
            )
        return mean_prefix, lse_prefix

    def pool(
        self,
        mean_prefix: torch.Tensor,
        lse_prefix: Optional[torch.Tensor],
        starts: torch.LongTensor,
        ends: torch.LongTensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        length = (ends - starts).clamp_min(1).unsqueeze(-1).float()
        span_sum = gather_prefix(mean_prefix, ends) - gather_prefix(mean_prefix, starts)
        pooled = span_sum / length
        if lse_prefix is not None:
            lse_end = gather_prefix(lse_prefix, ends)
            lse_start = gather_prefix(lse_prefix, starts)
            delta = (lse_start - lse_end).clamp(max=-1e-6)
            soft_max = lse_end + torch.log1p(-torch.exp(delta))
            soft_max = torch.nan_to_num(soft_max, neginf=0.0, posinf=0.0)
            pooled = torch.cat((pooled, soft_max), dim=-1)
        pooled = self.layer_norm(pooled.to(out_dtype))
        return self.dropout(pooled)


__all__ = ["SpanContentPooler", "gather_prefix"]
