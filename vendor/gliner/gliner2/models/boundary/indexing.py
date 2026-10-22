"""Memory-efficient batched indexing helpers for boundary tensors."""

from __future__ import annotations

import torch


def gather_states(
    states: torch.Tensor, indices: torch.LongTensor
) -> torch.Tensor:
    """Gather ``[B, N, D]`` at ``[B, Q, C]`` -> ``[B, Q, C, D]``.

    Flattening the query/candidate axes keeps autograd's gather-backward input
    at ``[B, N, D]`` instead of an expanded ``[B, Q, N, D]`` view.
    """
    b, n, dim = states.shape
    q, c = indices.shape[1:3]
    flat = indices.clamp(0, n - 1).reshape(b, q * c, 1).expand(b, q * c, dim)
    return states.gather(1, flat).view(b, q, c, dim)


def gather_rows(
    states: torch.Tensor, indices: torch.LongTensor
) -> torch.Tensor:
    """Gather ``[B, N, D]`` at ``[B, K]`` -> ``[B, K, D]``."""
    n = states.shape[1]
    dim = states.shape[-1]
    return states.gather(1, indices.clamp(0, n - 1).unsqueeze(-1).expand(-1, -1, dim))


__all__ = ["gather_rows", "gather_states"]
