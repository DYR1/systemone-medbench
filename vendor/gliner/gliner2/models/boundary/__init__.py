"""Sparse boundary architecture primitives.

Half-open ``[start, end)`` coordinates throughout. There is no width axis and
no production ``[L, L]`` or ``[L, W, D]`` tensor: proposal work streams over end
blocks and stays linear in sequence length for fixed schema/budgets.
"""

from gliner2.models.boundary.encoding import (
    BoundaryEncoding,
    BoundaryAttentionBlock,
    BoundaryEncoder,
    build_boundary_mask,
    shift_left_with_bos,
    shift_right_with_eos,
)
from gliner2.models.boundary.heads import BoundaryMarginals, BoundaryQueryHead
from gliner2.models.boundary.constants import MASK_LOGIT
from gliner2.models.boundary.indexing import gather_rows, gather_states
from gliner2.models.boundary.proposal import (
    ProposalSettings,
    ProposalStats,
    BoundaryProposals,
    SparseBoundaryProposer,
    assemble_candidates,
    resolve_boundary_budget,
)
from gliner2.models.boundary.content import SpanContentPooler, gather_prefix
from gliner2.models.boundary.rotary import RotaryBoundaryEmbedding
from gliner2.models.boundary.scoring import SparseBoundaryPairScorer
from gliner2.models.boundary.pool import (
    DocumentCandidatePool,
    PooledCandidates,
    SharedPoolScorer,
    OverlapBiasedCandidateAttention,
    classify_overlap_buckets,
)
from gliner2.models.boundary.losses import (
    asymmetric_focal_loss,
    balanced_multilabel_bce,
    build_candidate_labels,
    candidate_pair_loss,
    inside_consistency_loss,
    select_hard_negative_candidates,
    proposal_listwise_loss,
    reranker_listwise_loss,
    marginal_pair_consistency_loss,
    abstention_loss,
    count_log_rate_loss,
)
from gliner2.models.boundary.model import (
    BoundaryExtractorModel,
    BoundaryHead,
    decode_candidates,
    proposal_settings_from_head,
)
from gliner2.models.boundary.records import (
    FieldAssignmentScorer,
    InstanceCandidate,
    InstanceCandidateBatch,
    DecodedRecord,
    RecordGroupOutput,
    RecordHead,
    RecordSetDecoder,
    RecordSetOutput,
    create_anchor_instances,
    decode_group,
    derive_count,
)
from gliner2.models.boundary.relations import (
    RelationPairBatch,
    RelationProposalSettings,
    RelationTypeSpec,
    SparseRelationScorer,
    TypedRelationPairGenerator,
)

__all__ = [
    "MASK_LOGIT",
    "gather_rows",
    "gather_states",
    "BoundaryEncoding",
    "BoundaryAttentionBlock",
    "BoundaryEncoder",
    "build_boundary_mask",
    "shift_left_with_bos",
    "shift_right_with_eos",
    "BoundaryMarginals",
    "BoundaryQueryHead",
    "ProposalSettings",
    "ProposalStats",
    "BoundaryProposals",
    "SparseBoundaryProposer",
    "assemble_candidates",
    "resolve_boundary_budget",
    "SpanContentPooler",
    "gather_prefix",
    "RotaryBoundaryEmbedding",
    "SparseBoundaryPairScorer",
    "DocumentCandidatePool",
    "PooledCandidates",
    "SharedPoolScorer",
    "OverlapBiasedCandidateAttention",
    "classify_overlap_buckets",
    "asymmetric_focal_loss",
    "balanced_multilabel_bce",
    "build_candidate_labels",
    "candidate_pair_loss",
    "inside_consistency_loss",
    "select_hard_negative_candidates",
    "proposal_listwise_loss",
    "reranker_listwise_loss",
    "marginal_pair_consistency_loss",
    "abstention_loss",
    "count_log_rate_loss",
    "BoundaryExtractorModel",
    "BoundaryHead",
    "decode_candidates",
    "proposal_settings_from_head",
    "FieldAssignmentScorer",
    "InstanceCandidate",
    "InstanceCandidateBatch",
    "RecordGroupOutput",
    "RecordHead",
    "RecordSetDecoder",
    "RecordSetOutput",
    "create_anchor_instances",
    "DecodedRecord",
    "decode_group",
    "derive_count",
    "RelationPairBatch",
    "RelationProposalSettings",
    "RelationTypeSpec",
    "SparseRelationScorer",
    "TypedRelationPairGenerator",
]
