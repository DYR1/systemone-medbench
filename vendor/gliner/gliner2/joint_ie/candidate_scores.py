"""Architecture-neutral sparse candidate score format for joint IE.

The span architecture produces a dense width-oriented :class:`ScoreLattice`;
the boundary architecture produces sparse candidates directly. Both are mapped
onto :class:`CandidateScoreSet` — a flat list of mention scores plus optional
relation-role scores — which then feeds the *unchanged* ``NodeCandidate`` /
``EdgeCandidate`` / ``JointProblem`` optimizer contract. Coordinates are
half-open ``[start, end)`` token offsets throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, List, Mapping, Optional, Sequence, Tuple

from gliner2.joint_ie.candidates import (
    CandidateSource,
    EdgeCandidate,
    JointProblem,
    NodeCandidate,
    center_logit,
    sigmoid,
)


@dataclass(frozen=True)
class MentionScore:
    """One scored span mention (half-open ``[start, end)`` token offsets)."""

    query_id: int
    entity_type: str
    start: int
    end: int
    logit: float
    probability: float
    threshold: float = 0.5
    candidate_threshold: Optional[float] = None

    @property
    def key(self) -> Tuple[str, int, int]:
        return (self.entity_type, self.start, self.end)


@dataclass(frozen=True)
class RelationRoleScore:
    """A mention's compatibility with one role of one relation type."""

    relation_type: str
    role: str  # "head" | "tail"
    mention_id: Hashable
    logit: float
    probability: float


@dataclass(frozen=True)
class ScoredRelationEdge:
    """A scored (head, tail) relation proposal referencing mention keys."""

    relation_type: str
    head: Hashable
    tail: Hashable
    logit: float
    probability: float
    threshold: float = 0.5
    candidate_threshold: Optional[float] = None


@dataclass
class CandidateScoreSet:
    """Sparse, architecture-neutral candidate scores for one text."""

    text: str
    mentions: Tuple[MentionScore, ...]
    relation_roles: Tuple[RelationRoleScore, ...] = ()
    edges: Tuple[ScoredRelationEdge, ...] = ()
    classifications: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    text_tokens: Tuple[str, ...] = ()
    start_mappings: Tuple[int, ...] = ()
    end_mappings: Tuple[int, ...] = ()


def score_lattice_to_candidate_score_set(lattice: Any) -> CandidateScoreSet:
    """Convert a dense span :class:`ScoreLattice` into a sparse score set.

    Reads the entity task's single count hypothesis (``role_logits[0]`` shaped
    ``[num_types, L, W]``) and emits one :class:`MentionScore` per valid,
    above-floor span cell, mapping inclusive width cells to half-open spans.
    """
    span_starts = lattice.span_starts
    span_ends = lattice.span_ends
    valid = lattice.valid_span_mask

    mentions: List[MentionScore] = []
    query_id = 0
    for task in lattice.tasks:
        if task.task_type != "entities" or not task.count_hypotheses:
            continue
        hyp = task.count_hypotheses[0]
        role_logits = hyp.role_logits[0]           # [num_types, L, W]
        role_probs = hyp.role_probabilities[0]
        num_types = role_logits.shape[0]
        for t in range(num_types):
            entity_type = task.roles[t] if t < len(task.roles) else str(t)
            length = role_logits.shape[1]
            width = role_logits.shape[2]
            for i in range(length):
                for w in range(width):
                    if not bool(valid[i, w]):
                        continue
                    prob = float(role_probs[t, i, w])
                    start = int(span_starts[i, w])
                    end = int(span_ends[i, w]) + 1  # inclusive -> half-open
                    mentions.append(
                        MentionScore(
                            query_id=query_id,
                            entity_type=entity_type,
                            start=start,
                            end=end,
                            logit=float(role_logits[t, i, w]),
                            probability=prob,
                        )
                    )
            query_id += 1

    return CandidateScoreSet(
        text=lattice.text,
        mentions=tuple(mentions),
        text_tokens=tuple(getattr(lattice, "text_tokens", ())),
        start_mappings=tuple(getattr(lattice, "start_mappings", ())),
        end_mappings=tuple(getattr(lattice, "end_mappings", ())),
    )


def boundary_candidates_to_candidate_score_set(
    text: str,
    candidates: Any,
    query_specs: Sequence[Any],
    *,
    sample_index: int = 0,
    token_offset: int = 0,
    text_length: Optional[int] = None,
    pair_temperature: float = 1.0,
    entity_thresholds: Optional[Mapping[str, Optional[float]]] = None,
    entity_candidate_thresholds: Optional[
        Mapping[str, Optional[float]]
    ] = None,
    extra_mentions: Sequence[MentionScore] = (),
    edges: Sequence[ScoredRelationEdge] = (),
    text_tokens: Sequence[str] = (),
    start_mappings: Sequence[int] = (),
    end_mappings: Sequence[int] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> CandidateScoreSet:
    """Convert one boundary candidate batch row into sparse joint scores."""
    if pair_temperature <= 0:
        raise ValueError("pair_temperature must be positive")
    if text_length is None:
        text_length = len(start_mappings)
    thresholds = dict(entity_thresholds or {})
    candidate_thresholds = dict(entity_candidate_thresholds or {})
    best: dict[Tuple[str, int, int], MentionScore] = {
        mention.key: mention for mention in extra_mentions
    }

    for query_id, spec in enumerate(query_specs):
        task_type = (
            spec.get("task_type")
            if isinstance(spec, Mapping)
            else getattr(spec, "task_type", None)
        )
        if task_type != "entities" or query_id >= candidates.indices.shape[1]:
            continue
        entity_type = str(
            spec.get("field_name")
            if isinstance(spec, Mapping)
            else getattr(spec, "role_name", query_id)
        )
        threshold = thresholds.get(entity_type)
        threshold = 0.5 if threshold is None else float(threshold)
        valid = (
            candidates.valid_mask[sample_index, query_id]
            & candidates.query_mask[sample_index, query_id]
        )
        candidate_ids = valid.nonzero(as_tuple=False).flatten().tolist()
        for candidate_id in candidate_ids:
            start = int(
                candidates.indices[sample_index, query_id, candidate_id, 0]
            ) - token_offset
            end = int(
                candidates.indices[sample_index, query_id, candidate_id, 1]
            ) - token_offset
            if not (0 <= start < end <= int(text_length)):
                continue
            logit = float(
                candidates.pair_logits[
                    sample_index, query_id, candidate_id
                ].detach().float()
            ) / pair_temperature
            mention = MentionScore(
                query_id=query_id,
                entity_type=entity_type,
                start=start,
                end=end,
                logit=logit,
                probability=sigmoid(logit),
                threshold=threshold,
                candidate_threshold=candidate_thresholds.get(entity_type),
            )
            previous = best.get(mention.key)
            if previous is None or mention.logit > previous.logit:
                best[mention.key] = mention

    mentions = tuple(sorted(
        best.values(),
        key=lambda item: (
            item.entity_type, item.start, item.end, -item.logit
        ),
    ))
    return CandidateScoreSet(
        text=text,
        mentions=mentions,
        edges=tuple(edges),
        metadata=dict(metadata or {}),
        text_tokens=tuple(text_tokens),
        start_mappings=tuple(start_mappings),
        end_mappings=tuple(end_mappings),
    )


def candidate_score_set_to_problem(
    score_set: CandidateScoreSet,
    edges: Optional[Sequence[ScoredRelationEdge]] = None,
    *,
    mention_threshold: float = 0.5,
    constraints: Sequence[Any] = (),
    decision_threshold: float = 0.5,
    max_mentions_per_type: Optional[int] = None,
    max_mentions_by_type: Optional[Mapping[str, int]] = None,
    rescue_relation_endpoints: bool = False,
    edge_candidate_threshold: float = 0.0,
    max_edges_per_type: Optional[int] = None,
    entity_weight: float = 1.0,
    relation_weight: float = 1.0,
) -> JointProblem:
    """Build a :class:`JointProblem` from sparse mention + edge scores.

    Node/edge utilities are centered log-odds (positive => above threshold), so
    the existing greedy/beam optimizers and constraints work unchanged.
    """
    raw_edges = tuple(score_set.edges if edges is None else edges)
    edge_by_key: dict[Tuple[Any, ...], ScoredRelationEdge] = {}
    for edge in raw_edges:
        threshold = (
            edge_candidate_threshold
            if edge.candidate_threshold is None
            else edge.candidate_threshold
        )
        if edge.probability < threshold:
            continue
        key = (edge.relation_type, edge.head, edge.tail)
        previous = edge_by_key.get(key)
        if previous is None or edge.logit > previous.logit:
            edge_by_key[key] = edge
    edge_counts: dict[str, int] = {}
    retained_edges: List[ScoredRelationEdge] = []
    for edge in sorted(
        edge_by_key.values(),
        key=lambda item: (
            item.relation_type, -item.logit, str(item.head), str(item.tail)
        ),
    ):
        if (
            max_edges_per_type is not None
            and edge_counts.get(edge.relation_type, 0) >= max_edges_per_type
        ):
            continue
        retained_edges.append(edge)
        edge_counts[edge.relation_type] = (
            edge_counts.get(edge.relation_type, 0) + 1
        )
    relation_edges = tuple(retained_edges)
    rescue_ids = {
        endpoint
        for edge in relation_edges
        for endpoint in (edge.head, edge.tail)
    } if rescue_relation_endpoints else set()
    selected_mentions: List[MentionScore] = []
    per_type: dict[str, int] = {}
    type_limits = dict(max_mentions_by_type or {})
    for m in sorted(
        score_set.mentions,
        key=lambda item: (
            item.entity_type, -item.probability, item.start, item.end
        ),
    ):
        candidate_threshold = (
            mention_threshold
            if m.candidate_threshold is None
            else m.candidate_threshold
        )
        if m.probability < candidate_threshold and m.key not in rescue_ids:
            continue
        type_limit = type_limits.get(m.entity_type, max_mentions_per_type)
        if (
            type_limit is not None
            and per_type.get(m.entity_type, 0) >= type_limit
            and m.key not in rescue_ids
        ):
            continue
        selected_mentions.append(m)
        per_type[m.entity_type] = per_type.get(m.entity_type, 0) + 1

    nodes: List[NodeCandidate] = []
    keep_ids = set()
    for m in selected_mentions:
        candidate_threshold = (
            mention_threshold
            if m.candidate_threshold is None
            else m.candidate_threshold
        )
        node = NodeCandidate(
            entity_type=m.entity_type,
            start=m.start,
            end=m.end,
            score=entity_weight * center_logit(
                m.logit,
                m.threshold if m.threshold is not None else decision_threshold,
            ),
            probability=m.probability,
            source=(
                CandidateSource.RELATION_RESCUE
                if (
                    m.key in rescue_ids
                    and m.probability < candidate_threshold
                )
                else CandidateSource.ENTITY
            ),
            candidate_id=m.key,
        )
        nodes.append(node)
        keep_ids.add(m.key)

    edge_cands: List[EdgeCandidate] = []
    for edge_slot, e in enumerate(relation_edges):
        if e.head not in keep_ids or e.tail not in keep_ids:
            continue
        edge_cands.append(
            EdgeCandidate(
                relation_type=e.relation_type,
                head=e.head,
                tail=e.tail,
                score=relation_weight * center_logit(
                    e.logit,
                    e.threshold if e.threshold is not None else decision_threshold,
                ),
                head_probability=e.probability,
                tail_probability=e.probability,
                slot=edge_slot,
                hypothesis=e.relation_type,
            )
        )

    return JointProblem(
        nodes=tuple(nodes),
        edges=tuple(edge_cands),
        constraints=tuple(constraints),
    )


__all__ = [
    "MentionScore",
    "RelationRoleScore",
    "CandidateScoreSet",
    "ScoredRelationEdge",
    "score_lattice_to_candidate_score_set",
    "boundary_candidates_to_candidate_score_set",
    "candidate_score_set_to_problem",
]
