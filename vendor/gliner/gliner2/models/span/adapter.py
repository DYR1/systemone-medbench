"""Adapters bridging inclusive-span coordinates to half-open boundaries.

The span architecture represents spans as ``(token_start, token_end_inclusive)``
with a ``width = end - start`` axis. The boundary architecture and the shared
candidate contract use half-open ``[start, end)`` coordinates. These helpers
convert between the two so both architectures share one downstream convention.
"""

from __future__ import annotations

from typing import Sequence

import torch


def width_indices_to_boundaries(
    span_starts: torch.Tensor,
    span_ends_inclusive: torch.Tensor,
) -> torch.Tensor:
    """Convert inclusive end positions to half-open boundary coordinates.

    Args:
        span_starts: LongTensor of token start indices, shape ``[...]``.
        span_ends_inclusive: LongTensor of inclusive token end indices,
            same shape as ``span_starts``.

    Returns:
        LongTensor ``[..., 2]`` of half-open ``[start, end)`` pairs, where
        ``end = end_inclusive + 1``.
    """
    starts = span_starts.long()
    ends = span_ends_inclusive.long() + 1
    return torch.stack([starts, ends], dim=-1)


def inclusive_tokens_to_boundary_pair(start: int, end_inclusive: int):
    """Scalar variant: ``(start, end_inclusive) -> (start, end_inclusive + 1)``."""
    return start, end_inclusive + 1


def dense_span_scores_to_packed_candidates(
    logits: torch.Tensor,
    valid_span_mask: torch.BoolTensor,
    query_layouts: Sequence,
):
    """Convert dense span logits into a ``PackedCandidateBatch``.

    ``logits`` is the dense span score tensor with a width axis and
    ``valid_span_mask`` marks representable spans. This flattens the surviving
    ``(start, width)`` cells into half-open candidate pairs.

    This is a thin bridge used only where the span architecture must expose the
    shared candidate contract; the span model's production path is unchanged.
    """
    from gliner2.models.outputs import PackedCandidateBatch

    if logits.dim() != 4:
        raise ValueError(
            f"expected dense span logits [B, Q, L, W], got shape {tuple(logits.shape)}"
        )
    batch_size, num_queries, length, width = logits.shape
    device = logits.device

    batch_idx_list = []
    query_idx_list = []
    starts_list = []
    ends_list = []
    logit_list = []
    offsets = [0]

    for b in range(batch_size):
        for q in range(num_queries):
            mask_bq = valid_span_mask[b, q]
            positions = torch.nonzero(mask_bq, as_tuple=False)
            for pos in positions:
                start = int(pos[0].item())
                w = int(pos[1].item())
                batch_idx_list.append(b)
                query_idx_list.append(q)
                starts_list.append(start)
                ends_list.append(start + w + 1)
                logit_list.append(float(logits[b, q, start, w].item()))
            offsets.append(len(starts_list))

    long = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)
    flt = lambda xs: torch.tensor(xs, dtype=logits.dtype, device=device)

    return PackedCandidateBatch(
        batch_indices=long(batch_idx_list),
        query_indices=long(query_idx_list),
        starts=long(starts_list),
        ends=long(ends_list),
        proposal_logits=flt(logit_list),
        pair_logits=flt(logit_list),
        offsets=long(offsets),
        num_queries=num_queries,
    )
