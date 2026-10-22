"""Dense boundary targets reconstructed from sparse mention pairs."""

from __future__ import annotations

import torch


def dense_targets_from_pairs(
    pairs: torch.LongTensor,
    mask: torch.BoolTensor,
    text_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact start/end/inside targets on the pairs' device."""
    if pairs.shape[:-1] != mask.shape or pairs.shape[-1] != 2:
        raise ValueError(
            f"pairs {tuple(pairs.shape)} and mask {tuple(mask.shape)} "
            "are incompatible"
        )
    if text_length < 0:
        raise ValueError("text_length must be non-negative")
    b, q = pairs.shape[:2]
    valid = (
        mask
        & (pairs[..., 0] >= 0)
        & (pairs[..., 1] > pairs[..., 0])
        & (pairs[..., 1] <= text_length)
    )
    weights = valid.to(torch.float32)
    starts = pairs[..., 0].masked_fill(~mask, 0).clamp(0, text_length)
    ends = pairs[..., 1].masked_fill(~mask, 0).clamp(0, text_length)

    start_targets = torch.zeros(
        b, q, text_length + 1, dtype=torch.float32, device=pairs.device
    )
    end_targets = torch.zeros_like(start_targets)
    start_targets.scatter_add_(2, starts, weights).clamp_(max=1.0)
    end_targets.scatter_add_(2, ends, weights).clamp_(max=1.0)

    difference = torch.zeros(
        b, q, text_length + 2, dtype=torch.float32, device=pairs.device
    )
    difference.scatter_add_(2, starts, weights)
    difference.scatter_add_(2, ends, -weights)
    inside_targets = (
        difference[..., : text_length + 1].cumsum(-1)[..., :text_length] > 0.5
    ).to(torch.float32)
    return start_targets, end_targets, inside_targets


__all__ = ["dense_targets_from_pairs"]
