"""Sparse pair (candidate) reranking.

Produces one scalar logit per proposed candidate. The score is a sum of scalar
factors: start marginal, end marginal, endpoint compatibility, optional inside
evidence, continuous length features, and the proposal prior. No width
embedding table (which would reintroduce a maximum length) and no persistent
per-candidate vector is materialized beyond transient gathers.

Prior convention: the prior is the proposer's **marginal-free** endpoint
compatibility (``proposals.compat_logits``), so the start/end marginals enter
the score exactly once (via ``a``/``bmarg`` below). Do not use
``proposals.logits`` here — it already folds in the marginals and would
double-count them (Finding 7).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from gliner2.models.boundary.constants import MASK_LOGIT
from gliner2.models.boundary.proposal import BoundaryProposals
from gliner2.models.boundary.content import SpanContentPooler
from gliner2.models.boundary.indexing import gather_states
from gliner2.models.boundary.rotary import RotaryBoundaryEmbedding


def gather_boundary_states(boundary_states: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """Compatibility alias for :func:`gather_states`."""
    return gather_states(boundary_states, indices)


def interval_prefix_score(
    prefix: torch.Tensor,
    starts: torch.LongTensor,
    ends: torch.LongTensor,
    mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``prefix[..., end] - prefix[..., start]`` for ``[B, Q, C]`` indices.

    ``prefix`` is ``[B, Q, L+1]``. Returns ``[B, Q, C]``.
    """
    max_idx = prefix.shape[2] - 1
    starts = starts.clamp(0, max_idx)
    ends = ends.clamp(0, max_idx)
    p_end = torch.gather(prefix, 2, ends)
    p_start = torch.gather(prefix, 2, starts)
    interval = p_end - p_start
    if mean is not None:
        interval = interval + mean * (ends - starts).to(interval.dtype)
    return interval


def continuous_length_features(
    starts: torch.LongTensor,
    ends: torch.LongTensor,
    text_lengths: torch.LongTensor,
) -> torch.Tensor:
    """Length features ``[B, Q, C, 3]`` with no maximum-length lookup."""
    length = (ends - starts).clamp(min=1).float()
    b = starts.shape[0]
    tl = text_lengths.view(b, 1, 1).float().clamp(min=1)
    feats = torch.stack(
        [
            torch.log1p(length),
            length / tl,
            torch.rsqrt(length),
        ],
        dim=-1,
    )
    return feats


def mask_invalid_candidate_logits(logits: torch.Tensor, valid_mask: torch.BoolTensor) -> torch.Tensor:
    return logits.masked_fill(~valid_mask, MASK_LOGIT)


class SparseBoundaryPairScorer(nn.Module):
    def __init__(
        self,
        boundary_dim: int,
        query_dim: int,
        pair_dim: int,
        use_inside_evidence: bool = True,
        dropout: float = 0.1,
        enable_span_content: bool = False,
        content_dim: int = 64,
        content_soft_max_pool: bool = False,
        enable_rotary_endpoints: bool = False,
        rotary_base: float = 10000.0,
        query_conditioned_inside_weight: bool = False,
        endpoint_difference_features: bool = False,
        reranker_endpoint_compat: bool = True,
        multihead_pair_compat_heads: int = 1,
        content_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.boundary_dim = boundary_dim
        self.pair_dim = pair_dim
        self.use_inside_evidence = use_inside_evidence
        self.enable_span_content = enable_span_content
        self.enable_rotary_endpoints = enable_rotary_endpoints
        self.query_conditioned_inside_weight = query_conditioned_inside_weight
        self.endpoint_difference_features = endpoint_difference_features
        self.reranker_endpoint_compat = reranker_endpoint_compat
        if multihead_pair_compat_heads <= 0 or pair_dim % multihead_pair_compat_heads:
            raise ValueError(
                "pair_dim must be divisible by multihead_pair_compat_heads, got "
                f"{pair_dim} and {multihead_pair_compat_heads}"
            )
        self.multihead_pair_compat_heads = multihead_pair_compat_heads
        self.start_endpoint_projection = nn.Linear(boundary_dim, pair_dim)
        self.end_endpoint_projection = nn.Linear(boundary_dim, pair_dim)
        self.query_gate = nn.Linear(
            query_dim, pair_dim // 2 if enable_rotary_endpoints else pair_dim
        )
        self.length_query_projection = nn.Linear(query_dim, 3)
        self.inside_weight = (
            nn.Linear(query_dim, 1)
            if query_conditioned_inside_weight
            else nn.Parameter(torch.tensor(1.0))
        )
        self.endpoint_difference_projection = (
            nn.Linear(2 * pair_dim, 1) if endpoint_difference_features else None
        )
        self.rotary = (
            RotaryBoundaryEmbedding(pair_dim, rotary_base)
            if enable_rotary_endpoints else None
        )
        self.content_pooler = (
            SpanContentPooler(
                content_hidden_size if content_hidden_size is not None else query_dim,
                content_dim,
                dropout=dropout,
                use_soft_max_pool=content_soft_max_pool,
            )
            if enable_span_content else None
        )
        content_output_dim = (
            self.content_pooler.output_dim if self.content_pooler is not None else 0
        )
        self.content_query_projection = (
            nn.Linear(query_dim, content_output_dim)
            if enable_span_content else None
        )
        self.content_bias = (
            nn.Linear(content_output_dim, 1) if enable_span_content else None
        )
        self.dropout = nn.Dropout(dropout)
        # Keep this state-dict-changing layer last so heads=1 preserves the
        # historical initialization stream for every pre-existing parameter
        # and for caller-generated seeded inputs.
        rng_state = torch.random.get_rng_state()
        self.compat_mix = nn.Linear(multihead_pair_compat_heads, 1)
        nn.init.constant_(
            self.compat_mix.weight, 1.0 / multihead_pair_compat_heads
        )
        nn.init.zeros_(self.compat_mix.bias)
        torch.random.set_rng_state(rng_state)

    def project_endpoints(
        self, boundary_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and rotate all endpoints before candidate gathering."""
        start = self.start_endpoint_projection(boundary_states)
        end = self.end_endpoint_projection(boundary_states)
        if self.rotary is not None:
            n = boundary_states.shape[1]
            positions = torch.arange(n, device=boundary_states.device).view(1, n)
            start = self.rotary(start, positions)
            end = self.rotary(end, positions)
        return start, end

    def forward(
        self,
        boundary_states: torch.Tensor,      # [B, L+1, d]
        query_states: torch.Tensor,         # [B, Q, Hq]
        proposals: BoundaryProposals,
        start_logits: torch.Tensor,         # [B, Q, L+1]
        end_logits: torch.Tensor,           # [B, Q, L+1]
        inside_prefix: Optional[torch.Tensor],  # [B, Q, L+1] or None
        text_lengths: torch.LongTensor,     # [B]
        text_states: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.BoolTensor] = None,
        inside_prefix_mean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        starts = proposals.indices[..., 0]   # [B,Q,C]
        ends = proposals.indices[..., 1]
        valid = proposals.valid_mask
        scale = 1.0 / math.sqrt(self.pair_dim)

        # Endpoint compatibility.
        if (
            proposals.score_start_states is not None
            and proposals.score_end_states is not None
        ):
            s_proj = self.dropout(proposals.score_start_states)
            e_proj = self.dropout(proposals.score_end_states)
        else:
            # Compatibility path for external/legacy proposers.
            start_all, end_all = self.project_endpoints(boundary_states)
            s_proj = self.dropout(gather_boundary_states(start_all, starts))
            e_proj = self.dropout(gather_boundary_states(end_all, ends))
        gate = torch.sigmoid(self.query_gate(query_states))
        if self.enable_rotary_endpoints:
            gate = gate.repeat_interleave(2, dim=-1)
        gate = gate.unsqueeze(2)
        if self.reranker_endpoint_compat:
            per_head = (s_proj * gate * e_proj).reshape(
                *s_proj.shape[:-1], self.multihead_pair_compat_heads, -1
            ).sum(-1)
            compat = self.compat_mix(per_head).squeeze(-1) * scale
        else:
            compat = torch.zeros_like(starts, dtype=s_proj.dtype)
        if self.endpoint_difference_projection is not None:
            difference = torch.cat((s_proj - e_proj, (s_proj - e_proj).abs()), dim=-1)
            compat = compat + self.endpoint_difference_projection(difference).squeeze(-1)

        # Start/end marginals gathered at the candidate boundaries (added once).
        max_s = start_logits.shape[2] - 1
        max_e = end_logits.shape[2] - 1
        a = torch.gather(start_logits, 2, starts.clamp(0, max_s))
        bmarg = torch.gather(end_logits, 2, ends.clamp(0, max_e))

        # Prior is the proposer's marginal-free compatibility so marginals are
        # not double-counted. Fall back to the (masked) full prior only if a
        # legacy proposer did not populate compat_logits.
        prior_source = (
            proposals.compat_logits
            if proposals.compat_logits is not None else proposals.logits
        )
        if prior_source is None:
            raise ValueError("boundary proposals must provide compat_logits or logits")
        prior = torch.where(valid, prior_source, torch.zeros_like(prior_source))
        score = compat + a + bmarg + prior

        if self.content_pooler is not None:
            if text_states is None or text_mask is None:
                raise ValueError("span content scoring requires text_states and text_mask")
            mean_prefix, lse_prefix = self.content_pooler.build_prefix(text_states, text_mask)
            span_content = self.content_pooler.pool(
                mean_prefix, lse_prefix, starts, ends, score.dtype
            )
            coefficient = self.content_query_projection(query_states).unsqueeze(2)
            content_scale = 1.0 / math.sqrt(span_content.shape[-1])
            score = score + (span_content * coefficient).sum(-1) * content_scale
            score = score + self.content_bias(span_content).squeeze(-1)

        # Inside evidence.
        if self.use_inside_evidence and inside_prefix is not None:
            interval = interval_prefix_score(
                inside_prefix, starts, ends, inside_prefix_mean
            ).to(score.dtype)
            denom = torch.sqrt(
                (ends - starts).clamp(min=1).to(score.dtype)
            )
            inside_weight = (
                self.inside_weight(query_states).squeeze(-1).unsqueeze(-1)
                if self.query_conditioned_inside_weight
                else self.inside_weight
            )
            score = score + inside_weight * (interval / denom)

        # Length features.
        feats = continuous_length_features(starts, ends, text_lengths)     # [B,Q,C,3]
        length_coeff = self.length_query_projection(query_states).unsqueeze(2)  # [B,Q,1,3]
        length_score = (feats.to(length_coeff.dtype) * length_coeff).sum(-1)  # [B,Q,C]
        score = score + length_score

        return mask_invalid_candidate_logits(score, valid)
