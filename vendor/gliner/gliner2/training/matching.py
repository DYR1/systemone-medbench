"""Permutation-invariant record/event matching with an internal Hungarian solver.

The set decoders predict an unordered set of instances; training must match
predicted instances to gold instances by minimum assignment cost. We implement
a deterministic O(n^3) Hungarian algorithm here rather than depending on SciPy,
so results are reproducible and dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from gliner2.models.boundary.constants import MASK_LOGIT

_INF = float("inf")


def linear_sum_assignment(cost_matrix: torch.Tensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Minimum-cost assignment (Hungarian algorithm).

    Args:
        cost_matrix: ``[R, C]`` real costs.

    Returns:
        ``(row_ind, col_ind)`` long tensors of matched pairs, with ``row_ind``
        ascending. For rectangular inputs, ``min(R, C)`` pairs are returned.
    """
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2-D")
    # MPS does not support float64 tensors. Transfer first, then promote on CPU
    # for the numerically stable Hungarian solve.
    cost = cost_matrix.detach().cpu().to(torch.float64)
    if torch.isnan(cost).any():
        raise ValueError("cost_matrix contains NaN")
    # Map +/-inf to large finite sentinels so a fully-infinite row still admits
    # a (least-bad) augmenting column instead of failing the search.
    if torch.isinf(cost).any():
        finite = cost[torch.isfinite(cost)]
        scale = float(finite.abs().max()) if finite.numel() else 1.0
        big = 1e6 * (scale + 1.0)
        cost = torch.nan_to_num(cost, posinf=big, neginf=-big)
    r, c = cost.shape
    if r == 0 or c == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    try:
        from scipy.optimize import linear_sum_assignment as scipy_assignment
    except ImportError:
        scipy_assignment = None
    if scipy_assignment is not None:
        # SciPy's compiled solver avoids the Python O(n^2 m) hot loop. Add a
        # sub-ULP lexicographic offset so exact ties resolve reproducibly.
        array = cost.numpy()
        scale = max(float(abs(array).max()), 1.0)
        epsilon = torch.finfo(torch.float64).eps * scale
        tie = torch.arange(r * c, dtype=torch.float64).reshape(r, c).numpy()
        rows, cols = scipy_assignment(array + epsilon * tie)
        pairs = sorted(zip(rows.tolist(), cols.tolist()))
        return (
            torch.tensor([row for row, _ in pairs], dtype=torch.long),
            torch.tensor([col for _, col in pairs], dtype=torch.long),
        )

    transposed = c < r
    if transposed:
        cost = cost.t().contiguous()
    n, m = cost.shape  # n <= m

    # Jonker-Volgenant-style shortest augmenting path (O(n^2 m)).
    cost_list = cost.tolist()
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)   # p[j] = row assigned to column j (1-indexed), 0 = none
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [_INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost_list[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    rows: List[int] = []
    cols: List[int] = []
    for j in range(1, m + 1):
        if p[j] != 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    pairs = sorted(zip(rows, cols))
    row_ind = [a for a, _ in pairs]
    col_ind = [b for _, b in pairs]
    if transposed:
        # Undo transpose: swap roles and re-sort by row.
        pairs = sorted(zip(col_ind, row_ind))
        row_ind = [a for a, _ in pairs]
        col_ind = [b for _, b in pairs]
    return torch.tensor(row_ind, dtype=torch.long), torch.tensor(col_ind, dtype=torch.long)


@dataclass(frozen=True)
class MatchingResult:
    """Assignment of predicted instances (rows) to gold instances (cols)."""

    row_ind: torch.LongTensor
    col_ind: torch.LongTensor
    cost: float


def build_record_matching_cost(
    object_logits: torch.Tensor,      # [I]  object/no-object logit per instance query
    field_pointer_logits: torch.Tensor,  # [I, F, C]
    targets: Sequence[Sequence[int]],  # gold: list of instances, each = list of C-index per field (or -1)
) -> torch.Tensor:
    """Build an ``[I, G]`` negative-log-likelihood matching cost.

    Cost of assigning predicted instance ``i`` to gold instance ``g`` is the
    (negated) object log-prob plus the field-pointer log-probs at the gold
    candidate indices. Fields with gold index ``-1`` (absent) are skipped.
    """
    inst_queries = field_pointer_logits.shape[0]
    num_gold = len(targets)
    obj_logp = torch.nn.functional.logsigmoid(object_logits)          # [I]
    field_logp = torch.log_softmax(field_pointer_logits, dim=-1)      # [I, F, C]

    cost = torch.zeros(inst_queries, num_gold, dtype=field_logp.dtype)
    for g, inst in enumerate(targets):
        for i in range(inst_queries):
            score = obj_logp[i]
            for f, cidx in enumerate(inst):
                if cidx is None or cidx < 0:
                    continue
                score = score + field_logp[i, f, cidx]
            cost[i, g] = -score
    return cost


def match_record_instances(
    object_logits: torch.Tensor,
    field_pointer_logits: torch.Tensor,
    targets: Sequence[Sequence[int]],
) -> MatchingResult:
    """Match instance queries to gold instances by minimum assignment cost."""
    cost = build_record_matching_cost(object_logits, field_pointer_logits, targets)
    row_ind, col_ind = linear_sum_assignment(cost)
    total = float(cost[row_ind, col_ind].sum().detach()) if len(row_ind) else 0.0
    return MatchingResult(row_ind=row_ind, col_ind=col_ind, cost=total)


def build_dense_record_matching_cost(
    object_logits: torch.Tensor,       # [...,I]
    assign_logits: torch.Tensor,       # [...,I,F,1+C]
    gold_indicator: torch.BoolTensor,  # [...,N,F,C]
    scalar_fields: torch.BoolTensor,   # [...,F] or [F]
    instance_mask: torch.BoolTensor | None = None,  # [...,I]
) -> torch.Tensor:
    """Vectorized record cost ``[...,I,N]`` for dense document pools."""
    present = gold_indicator.any(-1)
    target = torch.cat(((~present).unsqueeze(-1), gold_indicator), -1)
    logp = F.log_softmax(assign_logits, -1)
    scalar_logprob = torch.logsumexp(
        logp.unsqueeze(-3).masked_fill(
            ~target.unsqueeze(-4), MASK_LOGIT
        ),
        -1,
    )
    candidates = assign_logits[..., 1:]
    list_logprob = -F.binary_cross_entropy_with_logits(
        candidates.unsqueeze(-3).expand(
            *candidates.shape[:-3],
            candidates.shape[-3],
            gold_indicator.shape[-3],
            candidates.shape[-2],
            candidates.shape[-1],
        ),
        gold_indicator.unsqueeze(-4).expand(
            *gold_indicator.shape[:-3],
            assign_logits.shape[-3],
            gold_indicator.shape[-3],
            gold_indicator.shape[-2],
            gold_indicator.shape[-1],
        ).to(candidates.dtype),
        reduction="none",
    ).sum(-1)
    field_logprob = torch.where(
        scalar_fields.unsqueeze(-2).unsqueeze(-2),
        scalar_logprob,
        list_logprob,
    ).sum(-1)
    cost = -(F.logsigmoid(object_logits).unsqueeze(-1) + field_logprob)
    if instance_mask is not None:
        cost = cost.masked_fill(~instance_mask.unsqueeze(-1), -MASK_LOGIT)
    return cost


__all__ = [
    "linear_sum_assignment",
    "MatchingResult",
    "build_record_matching_cost",
    "match_record_instances",
    "build_dense_record_matching_cost",
]
