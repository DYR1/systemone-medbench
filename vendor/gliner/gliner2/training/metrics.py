"""Boundary training metrics.

Proposal oracle recall is reported separately from final extraction F1 so a
failure can be attributed to proposal generation vs. reranking. All coordinates
are half-open ``[start, end)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from gliner2.models.outputs import CandidateTensorBatch
from gliner2.processing.targets import PaddedTargetBatch, TargetGraph

logger = logging.getLogger(__name__)


@dataclass
class BoundaryTrainingMetrics:
    total_loss: float = 0.0
    start_loss: float = 0.0
    end_loss: float = 0.0
    pair_loss: float = 0.0
    inside_loss: float = 0.0
    start_recall: float = 0.0
    end_recall: float = 0.0
    proposal_oracle_recall: float = 0.0
    exact_precision: float = 0.0
    exact_recall: float = 0.0
    exact_f1: float = 0.0
    candidate_count_per_query: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def candidate_oracle_recall(
    candidates: CandidateTensorBatch,
    targets: PaddedTargetBatch,
) -> float:
    """Fraction of gold mentions present among the proposed candidates.

    The comparison is fully tensorized and performs a single host transfer for
    the final scalar result.
    """
    candidate = candidates.indices.unsqueeze(2)       # [B,Q,1,C,2]
    gold = targets.mention_pairs.unsqueeze(3)         # [B,Q,G,1,2]
    same = (candidate == gold).all(-1)
    same = same & candidates.valid_mask.unsqueeze(2)
    hit = (same.any(-1) & targets.mention_mask).sum()
    total = targets.mention_mask.sum()
    total_value = int(total)
    if total_value == 0:
        logger.warning("candidate_oracle_recall: no gold mentions; reporting 0.0")
        return 0.0
    return float(hit) / total_value


def boundary_recall(
    marginal_logits: torch.Tensor,   # [B, Q, N]
    boundary_targets: torch.Tensor,  # [B, Q, N] (1.0 at gold boundaries)
    keep_mask: torch.BoolTensor,     # [B, Q, N]
    *,
    threshold: float = 0.0,
) -> float:
    """Recall of gold boundaries whose logit exceeds ``threshold``."""
    gold = (boundary_targets > 0.5) & keep_mask
    pred = (marginal_logits > threshold) & keep_mask
    tp = int((gold & pred).sum())
    total = int(gold.sum())
    if not total:
        logger.warning("boundary_recall: no gold boundaries; reporting 0.0")
        return 0.0
    return tp / total


def f1_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    *,
    zero_division: float = 0.0,
) -> Tuple[float, float, float]:
    """Precision/recall/F1 from counts.

    A zero denominator yields ``zero_division`` (default ``0.0``, matching the
    sklearn convention). This makes ``tp = fp = fn = 0`` score ``0.0`` rather
    than a misleading ``1.0`` that could promote a model producing nothing.
    """
    if (true_positive + false_positive):
        precision = true_positive / (true_positive + false_positive)
    else:
        logger.warning("f1_from_counts: no predictions; precision=%s", zero_division)
        precision = zero_division
    if (true_positive + false_negative):
        recall = true_positive / (true_positive + false_negative)
    else:
        logger.warning("f1_from_counts: no gold; recall=%s", zero_division)
        recall = zero_division
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def exact_span_counts(
    predicted: Sequence[List[List[Tuple[int, int]]]],
    gold: Sequence[List[set]],
) -> Tuple[int, int, int]:
    """``(tp, fp, fn)`` over exact ``(query, start, end)`` matches."""
    tp = fp = fn = 0
    for bi in range(len(predicted)):
        for qi in range(len(predicted[bi])):
            pred = set(predicted[bi][qi])
            g = gold[bi][qi] if bi < len(gold) and qi < len(gold[bi]) else set()
            tp += len(pred & g)
            fp += len(pred - g)
            fn += len(g - pred)
    return tp, fp, fn


def gold_from_target_graphs(
    graphs: Sequence[TargetGraph],
    query_count: int,
) -> List[List[set]]:
    """Build ``gold[b][q] = {(start, end)}`` from canonical target graphs."""
    out: List[List[set]] = []
    for graph in graphs:
        per_q: List[set] = [set() for _ in range(query_count)]
        for m in graph.mentions:
            if 0 <= m.query_id < query_count:
                per_q[m.query_id].add((m.start, m.end))
        out.append(per_q)
    return out


__all__ = [
    "BoundaryTrainingMetrics",
    "candidate_oracle_recall",
    "boundary_recall",
    "f1_from_counts",
    "exact_span_counts",
    "gold_from_target_graphs",
]
