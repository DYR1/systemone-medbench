"""Sparse boundary-pair proposal generation.

Selects a small set of start/end boundaries per query, then scores conditional
end (and, bidirectionally, start) boundaries in streaming blocks. Work is linear
in sequence length for fixed schema and budgets: the largest pair-score tensor
materialized at once is ``[B, Q, Ks, end_block_size]``. There is deliberately no
condition on ``end - start`` — a start at ``0`` may pair with an end at ``L``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gliner2.models.boundary.constants import MASK_LOGIT
from gliner2.models.boundary.indexing import gather_states
from gliner2.models.boundary.rotary import RotaryBoundaryEmbedding


# =============================================================================
# Settings and outputs
# =============================================================================

@dataclass(frozen=True)
class ProposalSettings:
    start_top_k: int
    end_top_k: int
    ends_per_start: int
    starts_per_end: int
    candidate_budget: int
    training_candidate_budget: int
    max_gold_per_query: int
    end_block_size: int
    bidirectional: bool = True
    export_mode: str = "auto"
    vectorized_pair_elements: int = 16_777_216
    enable_rotary_endpoints: bool = False
    rotary_base: float = 10000.0
    boundary_top_k_alpha: float = 0.0
    boundary_top_k_max: int = 128
    boundary_top_k_bucket: int = 8


@dataclass(frozen=True)
class ProposalStats:
    boundary_score_elements: int
    conditional_pair_score_elements: int
    max_materialized_pair_elements: int
    retained_candidate_count: torch.Tensor
    gold_hit_without_injection: Optional[torch.Tensor] = None
    gold_total: Optional[torch.Tensor] = None
    start_hit: Optional[torch.Tensor] = None
    end_hit: Optional[torch.Tensor] = None
    boundary_total: Optional[torch.Tensor] = None
    unique_candidates: Optional[torch.Tensor] = None


@dataclass
class BoundaryProposals:
    indices: torch.LongTensor          # [B, Q, C, 2] half-open [start, end)
    logits: Optional[torch.Tensor]      # [B, Q, C] full prior: compat + start/end marginals
    valid_mask: torch.BoolTensor       # [B, Q, C]
    gold_mask: Optional[torch.BoolTensor] = None  # [B, Q, C]
    stats: Optional[ProposalStats] = None
    # Marginal-free endpoint compatibility for the candidate (compat only, no
    # start/end marginals). The reranker uses this as its prior and adds the
    # marginals exactly once, so they are not double-counted (Finding 7).
    compat_logits: Optional[torch.Tensor] = None  # [B, Q, C]
    score_start_states: Optional[torch.Tensor] = None  # [B,Q,C,p]
    score_end_states: Optional[torch.Tensor] = None  # [B,Q,C,p]


# =============================================================================
# Boundary selection
# =============================================================================

def select_top_boundaries(
    logits: torch.Tensor,
    valid_mask: torch.BoolTensor,
    k: int,
) -> Tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
    """Select the top-``k`` boundaries by logit (stable, index tie-break).

    Args:
        logits: [B, Q, N]
        valid_mask: [B, Q, N] True where the boundary/query is valid.
        k: number to select.
    Returns:
        scores [B, Q, k], indices [B, Q, k], valid [B, Q, k].
    """
    n = logits.shape[-1]
    k = min(k, n)
    floor = MASK_LOGIT
    masked = logits.masked_fill(~valid_mask, floor)
    scores, idx = torch.sort(masked, dim=-1, descending=True, stable=True)
    scores = scores[..., :k]
    idx = idx[..., :k]
    valid = torch.gather(valid_mask, -1, idx)
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    return scores, idx, valid


def resolve_boundary_budget(
    n_boundaries: int,
    *,
    base_k: int,
    alpha: float,
    k_max: int,
    bucket: int,
) -> int:
    """Resolve a length-adaptive, bucketed boundary top-k from tensor shape."""
    if alpha <= 0.0:
        return base_k
    requested = int(math.ceil(alpha * max(n_boundaries - 1, 0)))
    requested = max(base_k, min(requested, k_max))
    return min(k_max, int(math.ceil(requested / bucket) * bucket))


def merge_running_topk(
    current_scores: torch.Tensor,
    current_indices: torch.LongTensor,
    block_scores: torch.Tensor,
    block_indices: torch.LongTensor,
    k: int,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Merge running top-k with stable score order.

    Exact-score ties retain concatenation order rather than the pre-PR-10
    boundary-index order. This is deterministic and intentionally rebaselined.
    """
    scores = torch.cat([current_scores, block_scores], dim=-1)
    indices = torch.cat([current_indices, block_indices], dim=-1)
    top_scores, order = torch.sort(scores, dim=-1, descending=True, stable=True)
    take = min(k, scores.shape[-1])
    top_scores = top_scores[..., :take]
    order = order[..., :take]
    top_indices = torch.gather(indices, -1, order)
    return top_scores, top_indices


def _score_ends_blockwise(
    sq: torch.Tensor,                  # [B, Q, K, d] projected+gated start states
    end_proj_all: torch.Tensor,        # [B, L+1, d] projected end states
    start_indices: torch.LongTensor,   # [B, Q, K]
    start_scores: torch.Tensor,        # [B, Q, K]
    start_valid: torch.BoolTensor,     # [B, Q, K] True where the start slot is real
    boundary_mask: torch.BoolTensor,   # [B, L+1]
    query_mask: torch.BoolTensor,      # [B, Q]
    end_marginals: torch.Tensor,       # [B, Q, L+1]
    block_size: int,
    top_k: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.LongTensor, int, int]:
    """Stream over end blocks, keeping running top-``top_k`` ends per start.

    Returns ``(top_scores, top_end_idx, conditional_elems, max_block_elems)``
    where scores/idx are ``[B, Q, K, top_k]``.
    """
    b, q, k, d = sq.shape
    n = end_proj_all.shape[1]
    device = sq.device
    neg_inf = MASK_LOGIT

    top_scores = torch.full((b, q, k, top_k), neg_inf, device=device, dtype=sq.dtype)
    top_idx = torch.zeros((b, q, k, top_k), device=device, dtype=torch.long)

    conditional_elems = 0
    max_block_elems = 0

    for j0 in range(0, n, block_size):
        j1 = min(j0 + block_size, n)
        e = j1 - j0
        ej = end_proj_all[:, j0:j1]                                    # [B, e, d]
        block_compat = torch.einsum("bqkd,bed->bqke", sq, ej) * scale  # [B,Q,K,e]
        end_marg_block = end_marginals[:, :, j0:j1].unsqueeze(2)        # [B,Q,1,e]
        block = block_compat + end_marg_block + start_scores.unsqueeze(-1)

        end_index = torch.arange(j0, j1, device=device)               # [e]
        # valid end iff boundary valid, query valid, end > start, and the start
        # slot itself is real. Dead start slots (padded top-k, score sentinel 0)
        # must not spawn candidates for a short sample batched with a long one.
        end_valid = boundary_mask[:, j0:j1].view(b, 1, 1, e)          # [B,1,1,e]
        q_valid = query_mask.view(b, q, 1, 1)
        after_start = end_index.view(1, 1, 1, e) > start_indices.unsqueeze(-1)
        keep = end_valid & q_valid & after_start & start_valid.unsqueeze(-1)
        block = block.masked_fill(~keep, neg_inf)

        block_idx = end_index.view(1, 1, 1, e).expand(b, q, k, e)
        top_scores, top_idx = merge_running_topk(top_scores, top_idx, block, block_idx, top_k)

        conditional_elems += b * q * k * e
        max_block_elems = max(max_block_elems, b * q * k * e)

    return top_scores, top_idx, conditional_elems, max_block_elems


def _score_starts_blockwise(
    eq: torch.Tensor,                  # [B, Q, K, d] projected+gated end states
    start_proj_all: torch.Tensor,      # [B, L+1, d]
    end_indices: torch.LongTensor,     # [B, Q, K]
    end_scores: torch.Tensor,          # [B, Q, K]
    end_valid: torch.BoolTensor,       # [B, Q, K] True where the end slot is real
    boundary_mask: torch.BoolTensor,
    query_mask: torch.BoolTensor,
    start_marginals: torch.Tensor,     # [B, Q, L+1]
    block_size: int,
    top_k: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.LongTensor, int, int]:
    """Bidirectional counterpart: top starts strictly before each selected end."""
    b, q, k, d = eq.shape
    n = start_proj_all.shape[1]
    device = eq.device
    neg_inf = MASK_LOGIT

    top_scores = torch.full((b, q, k, top_k), neg_inf, device=device, dtype=eq.dtype)
    top_idx = torch.zeros((b, q, k, top_k), device=device, dtype=torch.long)

    conditional_elems = 0
    max_block_elems = 0

    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        e = i1 - i0
        sj = start_proj_all[:, i0:i1]
        block_compat = torch.einsum("bqkd,bed->bqke", eq, sj) * scale
        start_marg_block = start_marginals[:, :, i0:i1].unsqueeze(2)
        block = block_compat + start_marg_block + end_scores.unsqueeze(-1)

        start_index = torch.arange(i0, i1, device=device)
        b_valid = boundary_mask[:, i0:i1].view(b, 1, 1, e)
        q_valid = query_mask.view(b, q, 1, 1)
        before_end = start_index.view(1, 1, 1, e) < end_indices.unsqueeze(-1)
        # Symmetric to the forward direction: a dead end slot must not spawn
        # start candidates.
        keep = b_valid & q_valid & before_end & end_valid.unsqueeze(-1)
        block = block.masked_fill(~keep, neg_inf)

        block_idx = start_index.view(1, 1, 1, e).expand(b, q, k, e)
        top_scores, top_idx = merge_running_topk(top_scores, top_idx, block, block_idx, top_k)

        conditional_elems += b * q * k * e
        max_block_elems = max(max_block_elems, b * q * k * e)

    return top_scores, top_idx, conditional_elems, max_block_elems


# =============================================================================
# Deduplication, gold injection, padding (per (b, q))
# =============================================================================

def _stable_desc_order(scores: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """Order indices by descending score, ties broken by (start, end) ascending."""
    n = scores.shape[0]
    device = scores.device
    # Composite sort: primary score desc, then start asc, then end asc.
    # Sort ascending by end, then start, then by -score using stable sorts.
    order = torch.arange(n, device=device)
    # end asc
    order = order[torch.argsort(ends[order], stable=True)]
    # start asc
    order = order[torch.argsort(starts[order], stable=True)]
    # score desc (stable keeps prior tie-break)
    order = order[torch.argsort(-scores[order], stable=True)]
    return order


def deduplicate_boundary_pairs(
    starts: torch.Tensor,
    ends: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.BoolTensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse duplicate ``(start, end)`` pairs keeping the highest score.

    Test/reference helper only; production uses :func:`assemble_candidates`,
    which is fully batched and has no data-dependent host synchronization.
    Operates on 1-D tensors for a single (sample, query). Returns
    ``(starts, ends, scores)`` of unique valid pairs, ordered by descending
    score with deterministic tie-break ``(start asc, end asc)``.

    Fully vectorized: no per-candidate Python/scalar synchronization. Duplicate
    collapse relies on the total order below — the first occurrence of each
    ``(start, end)`` key in score-descending order is its highest-scoring copy,
    so keeping first occurrences both deduplicates and preserves the final order.
    """
    device = starts.device
    empty_l = torch.zeros(0, dtype=torch.long, device=device)
    empty_f = torch.zeros(0, dtype=scores.dtype, device=device)
    if starts.numel() == 0 or not bool(valid.any()):
        return empty_l, empty_l, empty_f

    valid = valid.bool()
    s = starts[valid].to(torch.long)
    e = ends[valid].to(torch.long)
    sc = scores[valid]

    # Total order: score desc, then (start, end) ascending.
    order = _stable_desc_order(sc, s, e)
    s, e, sc = s[order], e[order], sc[order]

    # Collapse duplicate (start, end) rows keeping the first (highest-scoring)
    # occurrence. ``first[inverse]`` is the earliest ordered position per key.
    pairs = torch.stack((s, e), dim=1)                       # [m, 2]
    _, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    m = s.shape[0]
    positions = torch.arange(m, device=device)
    # ``inverse`` ids are in ``[0, num_unique)`` and ``num_unique <= m``, so a
    # length-``m`` buffer safely covers every key without a scalar sync.
    first = positions.new_full((m,), m)
    first = first.scatter_reduce(0, inverse, positions, reduce="amin", include_self=True)
    keep = positions == first[inverse]
    return s[keep], e[keep], sc[keep]


def assemble_candidates(
    pair_starts: torch.LongTensor,
    pair_ends: torch.LongTensor,
    pair_scores: torch.Tensor,
    pair_valid: torch.BoolTensor,
    query_mask: torch.BoolTensor,
    *,
    capacity: int,
    n_boundaries: int,
    gold_pairs: Optional[torch.LongTensor] = None,
    gold_mask: Optional[torch.BoolTensor] = None,
    gold_injection_prob: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[
    torch.LongTensor,
    torch.BoolTensor,
    torch.BoolTensor,
    torch.LongTensor,
    torch.BoolTensor,
]:
    """Batched candidate deduplication, gold injection, and capacity trimming.

    Candidate identity is encoded as one int64 key. The output total order is
    descending selection score with an ascending ``(start, end)`` tie-break.
    ``pre_keys``/``pre_valid`` describe proposals before gold injection and are
    returned for optional recall diagnostics.
    """
    dtype = pair_scores.dtype
    floor = MASK_LOGIT
    ceiling = -floor
    invalid_key = n_boundaries * n_boundaries

    pre_valid = pair_valid & query_mask.unsqueeze(-1)
    pre_keys = pair_starts * n_boundaries + pair_ends
    pre_keys = torch.where(
        pre_valid, pre_keys, torch.full_like(pre_keys, invalid_key)
    )
    keys = pre_keys
    scores = torch.where(
        pre_valid, pair_scores, torch.full_like(pair_scores, floor)
    )
    valid = pre_valid
    is_gold = torch.zeros_like(valid)

    if gold_pairs is not None and gold_mask is not None:
        gvalid = gold_mask & query_mask.unsqueeze(-1)
        oob = (gold_pairs[..., 0] >= n_boundaries) | (gold_pairs[..., 1] >= n_boundaries) | (gold_pairs < 0).any(-1)
        gvalid = gvalid & ~oob
        if gold_injection_prob <= 0.0:
            gvalid = torch.zeros_like(gvalid)
        elif gold_injection_prob < 1.0:
            sampled = torch.rand(
                gvalid.shape, device=gvalid.device, generator=generator
            )
            gvalid = gvalid & (sampled < gold_injection_prob)
        safe_gold = gold_pairs.clamp(0, n_boundaries - 1)
        gkeys = safe_gold[..., 0] * n_boundaries + safe_gold[..., 1]
        gkeys = torch.where(
            gvalid, gkeys, torch.full_like(gkeys, invalid_key)
        )
        gscores = torch.where(
            gvalid,
            torch.full(gvalid.shape, ceiling, dtype=dtype, device=pair_scores.device),
            torch.full(gvalid.shape, floor, dtype=dtype, device=pair_scores.device),
        )
        keys = torch.cat((keys, gkeys), dim=-1)
        scores = torch.cat((scores, gscores), dim=-1)
        valid = torch.cat((valid, gvalid), dim=-1)
        is_gold = torch.cat((is_gold, gvalid), dim=-1)

    # Group equal keys while keeping the highest-scoring occurrence first.
    # Unlike merge_running_topk, assemble keeps the historical score-desc /
    # key-ascending order; PR-10 permits a tie-order change only in the merge.
    by_score = torch.argsort(scores, dim=-1, descending=True, stable=True)
    keys = torch.gather(keys, -1, by_score)
    scores = torch.gather(scores, -1, by_score)
    valid = torch.gather(valid, -1, by_score)
    is_gold = torch.gather(is_gold, -1, by_score)
    by_key = torch.argsort(keys, dim=-1, stable=True)
    keys = torch.gather(keys, -1, by_key)
    scores = torch.gather(scores, -1, by_key)
    valid = torch.gather(valid, -1, by_key)
    is_gold = torch.gather(is_gold, -1, by_key)

    first = torch.ones_like(valid)
    first[..., 1:] = keys[..., 1:] != keys[..., :-1]
    keep = valid & first
    scores = torch.where(keep, scores, torch.full_like(scores, floor))
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    take = min(capacity, order.shape[-1])
    order = order[..., :take]
    selected_keys = torch.gather(keys, -1, order)
    selected_valid = torch.gather(keep, -1, order)
    selected_gold = torch.gather(is_gold, -1, order) & selected_valid

    starts = torch.div(selected_keys, n_boundaries, rounding_mode="floor")
    ends = selected_keys - starts * n_boundaries
    indices = torch.stack((starts, ends), dim=-1)
    indices = torch.where(selected_valid.unsqueeze(-1), indices, torch.zeros_like(indices))

    if take < capacity:
        pad = capacity - take
        indices = F.pad(indices, (0, 0, 0, pad))
        selected_valid = F.pad(selected_valid, (0, pad), value=False)
        selected_gold = F.pad(selected_gold, (0, pad), value=False)
    return indices, selected_valid, selected_gold, pre_keys, pre_valid


class SparseBoundaryProposer(nn.Module):
    def __init__(self, boundary_dim: int, query_dim: int, settings: ProposalSettings):
        super().__init__()
        self.boundary_dim = boundary_dim
        self.settings = settings
        self.start_pair_projection = nn.Linear(boundary_dim, boundary_dim)
        self.end_key_projection = nn.Linear(boundary_dim, boundary_dim)
        self.start_query_projection = nn.Linear(
            query_dim, boundary_dim // 2 if settings.enable_rotary_endpoints else boundary_dim
        )
        self.rotary = (
            RotaryBoundaryEmbedding(boundary_dim, settings.rotary_base)
            if settings.enable_rotary_endpoints else None
        )

    def score_explicit_pairs(
        self,
        boundary_states: torch.Tensor,
        query_states: torch.Tensor,
        indices: torch.LongTensor,
        valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Return the marginal-free proposal prior for explicit span pairs.

        This is the differentiable compatibility term used for candidates
        selected by :meth:`forward`, evaluated at caller-provided half-open
        ``[start, end)`` coordinates.  It lets downstream decoders force-score
        a retained span for another query without depending on that query's
        sparse top-k proposal set.
        """
        if indices.dim() != 4 or indices.shape[-1] != 2:
            raise ValueError(
                f"indices must be [B, Q, C, 2], got {tuple(indices.shape)}"
            )
        if valid_mask.shape != indices.shape[:-1]:
            raise ValueError(
                "valid_mask must match indices [B, Q, C], got "
                f"{tuple(valid_mask.shape)} and {tuple(indices.shape)}"
            )
        if (
            indices.shape[0] != boundary_states.shape[0]
            or indices.shape[0] != query_states.shape[0]
            or indices.shape[1] != query_states.shape[1]
        ):
            raise ValueError("explicit pair batch/query dimensions do not match states")

        start_all = self.start_pair_projection(boundary_states)
        end_all = self.end_key_projection(boundary_states)
        if self.rotary is not None:
            positions = torch.arange(
                boundary_states.shape[1], device=boundary_states.device
            ).view(1, -1)
            start_all = self.rotary(start_all, positions)
            end_all = self.rotary(end_all, positions)
        gate = torch.sigmoid(self.start_query_projection(query_states))
        if self.rotary is not None:
            gate = gate.repeat_interleave(2, dim=-1)

        starts = indices[..., 0].clamp(0, boundary_states.shape[1] - 1)
        ends = indices[..., 1].clamp(0, boundary_states.shape[1] - 1)
        start_states = gather_states(start_all, starts) * gate.unsqueeze(2)
        end_states = gather_states(end_all, ends)
        compatibility = (
            (start_states * end_states).sum(-1)
            / math.sqrt(self.boundary_dim)
        )
        return torch.where(
            valid_mask, compatibility, torch.zeros_like(compatibility)
        )

    def forward(
        self,
        boundary_states: torch.Tensor,   # [B, L+1, d]
        boundary_mask: torch.BoolTensor,  # [B, L+1]
        query_states: torch.Tensor,       # [B, Q, Hq]
        query_mask: torch.BoolTensor,     # [B, Q]
        start_logits: torch.Tensor,       # [B, Q, L+1]
        end_logits: torch.Tensor,         # [B, Q, L+1]
        *,
        gold_pairs: Optional[torch.LongTensor] = None,   # [B, Q, G, 2]
        gold_mask: Optional[torch.BoolTensor] = None,     # [B, Q, G]
        return_stats: bool = False,
        return_proposal_logits: bool = True,
        gold_injection_prob: float = 1.0,
        generator: Optional[torch.Generator] = None,
        scorer_start_states: Optional[torch.Tensor] = None,  # [B,L+1,p]
        scorer_end_states: Optional[torch.Tensor] = None,  # [B,L+1,p]
    ) -> BoundaryProposals:
        s = self.settings
        b, n, d = boundary_states.shape
        q = query_states.shape[1]
        device = boundary_states.device
        scale = 1.0 / math.sqrt(self.boundary_dim)
        training = self.training and gold_pairs is not None
        capacity = s.training_candidate_budget if training else s.candidate_budget
        start_k = resolve_boundary_budget(
            n,
            base_k=s.start_top_k,
            alpha=s.boundary_top_k_alpha,
            k_max=s.boundary_top_k_max,
            bucket=s.boundary_top_k_bucket,
        )
        end_k = resolve_boundary_budget(
            n,
            base_k=s.end_top_k,
            alpha=s.boundary_top_k_alpha,
            k_max=s.boundary_top_k_max,
            bucket=s.boundary_top_k_bucket,
        )

        # Export mode materializes a single full-width [B,Q,Ks,L+1] block
        # (still linear in L for fixed Ks) instead of streaming blocks — this
        # avoids the block loop for graph exporters while keeping identical
        # results and never building an [L, L] tensor.
        pair_elements = b * q * start_k * n
        use_vectorized = (
            s.export_mode == "vectorized"
            or (
                s.export_mode == "auto"
                and pair_elements <= s.vectorized_pair_elements
            )
        )
        end_block = n if use_vectorized else s.end_block_size

        b_valid = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)  # [B,Q,L+1]

        # Projected boundary states (shared across queries).
        start_proj_all = self.start_pair_projection(boundary_states)  # [B,L+1,d]
        end_proj_all = self.end_key_projection(boundary_states)       # [B,L+1,d]
        if self.rotary is not None:
            positions = torch.arange(n, device=device).view(1, n)
            start_proj_all = self.rotary(start_proj_all, positions)
            end_proj_all = self.rotary(end_proj_all, positions)
        gate = torch.sigmoid(self.start_query_projection(query_states))  # [B,Q,d]
        if self.rotary is not None:
            gate = gate.repeat_interleave(2, dim=-1)

        # Selection is deliberately non-differentiable. The selected endpoints
        # are rescored below from the live projections, preserving the only
        # default gradient path into all proposer projections.
        select_start = start_proj_all.detach()
        select_end = end_proj_all.detach()
        select_gate = gate.detach()
        select_start_logits = start_logits.detach()
        select_end_logits = end_logits.detach()

        # ---- forward direction: top starts -> conditional ends --------------
        with torch.no_grad():
            st_scores, st_idx, st_valid = select_top_boundaries(
                select_start_logits, b_valid, start_k
            )
            # gather start states and gate by query
            st_states = gather_states(select_start, st_idx)  # [B,Q,Ks,d]
            sq = st_states * select_gate.unsqueeze(2)
            fwd_scores, fwd_end_idx, cond_e1, maxe1 = _score_ends_blockwise(
                sq, select_end, st_idx, st_scores, st_valid,
                boundary_mask, query_mask, select_end_logits, end_block,
                s.ends_per_start, scale,
            )
        # forward pairs: (start = st_idx[k], end = fwd_end_idx[k, :])
        fwd_start = st_idx.unsqueeze(-1).expand(-1, -1, -1, s.ends_per_start)  # [B,Q,Ks,eps]
        fwd_pairs_s = fwd_start.reshape(b, q, -1)
        fwd_pairs_e = fwd_end_idx.reshape(b, q, -1)
        fwd_pairs_sc = fwd_scores.reshape(b, q, -1)
        fwd_pairs_valid = (
            st_valid.unsqueeze(-1)
            & query_mask.view(b, q, 1, 1)
            & boundary_mask.gather(
                1, fwd_end_idx.reshape(b, -1).clamp(0, boundary_mask.shape[1] - 1)
            ).view_as(fwd_end_idx)
            & (fwd_end_idx > fwd_start)
        ).reshape(b, q, -1)

        pair_starts = [fwd_pairs_s]
        pair_ends = [fwd_pairs_e]
        pair_scores = [fwd_pairs_sc]
        pair_valid = [fwd_pairs_valid]

        cond_e2 = 0
        maxe2 = 0
        if s.bidirectional:
            with torch.no_grad():
                en_scores, en_idx, en_valid = select_top_boundaries(
                    select_end_logits, b_valid, end_k
                )
                en_states = gather_states(select_end, en_idx)
                eq = en_states * select_gate.unsqueeze(2)
                bwd_scores, bwd_start_idx, cond_e2, maxe2 = _score_starts_blockwise(
                    eq, select_start, en_idx, en_scores, en_valid,
                    boundary_mask, query_mask, select_start_logits, end_block,
                    s.starts_per_end, scale,
                )
            bwd_end = en_idx.unsqueeze(-1).expand(-1, -1, -1, s.starts_per_end)
            pair_starts.append(bwd_start_idx.reshape(b, q, -1))
            pair_ends.append(bwd_end.reshape(b, q, -1))
            pair_scores.append(bwd_scores.reshape(b, q, -1))
            pair_valid.append(
                (
                    en_valid.unsqueeze(-1)
                    & query_mask.view(b, q, 1, 1)
                    & boundary_mask.gather(
                        1, bwd_start_idx.reshape(b, -1).clamp(0, boundary_mask.shape[1] - 1)
                    ).view_as(bwd_start_idx)
                    & (bwd_end > bwd_start_idx)
                ).reshape(b, q, -1)
            )

        all_s = torch.cat(pair_starts, dim=-1)   # [B,Q,P]
        all_e = torch.cat(pair_ends, dim=-1)
        # Ordering/selection scores only; differentiable logits are recomputed
        # from the chosen indices below, so detach to avoid grad-to-scalar noise.
        all_sc = torch.cat(pair_scores, dim=-1).detach()
        all_valid = torch.cat(pair_valid, dim=-1)

        # Batched deduplication and optional gold injection. Gold tensors are
        # still available during eval for diagnostics, but are only injected
        # while training.
        with torch.no_grad():
            out_idx, out_valid, out_gold, pre_keys, pre_valid = assemble_candidates(
                all_s,
                all_e,
                all_sc,
                all_valid,
                query_mask,
                capacity=capacity,
                n_boundaries=n,
                gold_pairs=gold_pairs if training else None,
                gold_mask=gold_mask if training else None,
                gold_injection_prob=gold_injection_prob,
                generator=generator,
            )

        # Differentiable proposal logits: recompute from the (detached) selected
        # indices so gradients flow to the proposer projections and marginals.
        si = out_idx[..., 0]                                            # [B,Q,C]
        ej = out_idx[..., 1]
        score_start_selected = score_end_selected = None
        if scorer_start_states is not None and scorer_end_states is not None:
            prop_dim = start_proj_all.shape[-1]
            score_dim = scorer_start_states.shape[-1]
            start_all = torch.cat((start_proj_all, scorer_start_states), dim=-1)
            end_all = torch.cat((end_proj_all, scorer_end_states), dim=-1)
            g_s, score_start_selected = gather_states(start_all, si).split(
                (prop_dim, score_dim), dim=-1
            )
            g_e, score_end_selected = gather_states(end_all, ej).split(
                (prop_dim, score_dim), dim=-1
            )
        else:
            g_s = gather_states(start_proj_all, si)
            g_e = gather_states(end_proj_all, ej)
        g_s = g_s * gate.unsqueeze(2)
        compat = (g_s * g_e).sum(-1) * scale                            # [B,Q,C]
        out_logits = None
        if self.training or return_proposal_logits:
            sm = torch.gather(start_logits, 2, si.clamp(0, start_logits.shape[2] - 1))
            em = torch.gather(end_logits, 2, ej.clamp(0, end_logits.shape[2] - 1))
            logits_diff = compat + sm + em
            neg_inf_like = torch.full_like(logits_diff, MASK_LOGIT)
            out_logits = torch.where(out_valid, logits_diff, neg_inf_like)
        # Marginal-free compatibility prior for the reranker (marginals are added
        # once inside the scorer). Masked to zero on invalid slots so it can be
        # summed directly.
        out_compat = torch.where(out_valid, compat, torch.zeros_like(compat))

        stats = None
        if return_stats:
            retained = out_valid.sum()
            gold_hit = gold_total = start_hit = end_hit = boundary_total = None
            if gold_pairs is not None and gold_mask is not None:
                diagnostic_gold = gold_mask & query_mask.unsqueeze(-1)
                gold_keys = gold_pairs[..., 0] * n + gold_pairs[..., 1]
                pair_hit = (
                    (gold_keys.unsqueeze(-1) == pre_keys.unsqueeze(-2))
                    & pre_valid.unsqueeze(-2)
                ).any(-1) & diagnostic_gold
                gold_hit = pair_hit.sum()
                gold_total = diagnostic_gold.sum()

                selected_starts = st_idx
                selected_starts_valid = st_valid
                if s.bidirectional:
                    selected_ends = en_idx
                    selected_ends_valid = en_valid
                else:
                    selected_ends = fwd_end_idx.reshape(b, q, -1)
                    selected_ends_valid = fwd_pairs_valid
                start_hit = (
                    (gold_pairs[..., 0].unsqueeze(-1) == selected_starts.unsqueeze(-2))
                    & selected_starts_valid.unsqueeze(-2)
                ).any(-1) & diagnostic_gold
                end_hit = (
                    (gold_pairs[..., 1].unsqueeze(-1) == selected_ends.unsqueeze(-2))
                    & selected_ends_valid.unsqueeze(-2)
                ).any(-1) & diagnostic_gold
                start_hit = start_hit.sum()
                end_hit = end_hit.sum()
                boundary_total = diagnostic_gold.sum()
            stats = ProposalStats(
                boundary_score_elements=b * q * n * 2,
                conditional_pair_score_elements=cond_e1 + cond_e2,
                max_materialized_pair_elements=max(maxe1, maxe2),
                retained_candidate_count=retained,
                gold_hit_without_injection=gold_hit,
                gold_total=gold_total,
                start_hit=start_hit,
                end_hit=end_hit,
                boundary_total=boundary_total,
                unique_candidates=retained,
            )

        return BoundaryProposals(
            indices=out_idx,
            logits=out_logits,
            valid_mask=out_valid,
            gold_mask=out_gold if training else None,
            stats=stats,
            compat_logits=out_compat,
            score_start_states=score_start_selected,
            score_end_states=score_end_selected,
        )
