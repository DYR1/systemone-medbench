"""Shared optimizer contracts and constraint dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Hashable, Iterable, List, Sequence, Tuple

from ..candidates import EdgeCandidate, JointProblem, NodeCandidate
from ..constraints import InverseRelation, SymmetricRelation


@dataclass(frozen=True)
class JointSolution:
    nodes: Tuple[NodeCandidate, ...]
    edges: Tuple[EdgeCandidate, ...]
    score: float
    # False signals that the decoder could not produce a constraint-satisfying
    # non-trivial assignment and fell back (e.g. to the empty solution). A
    # feasible solution — including the empty one produced when there is nothing
    # to extract — is True.
    feasible: bool = True

    @property
    def node_ids(self) -> FrozenSet[Hashable]:
        return frozenset(node.candidate_id for node in self.nodes)


class BaseOptimizer:
    """Base class with duck-typed constraint handling."""

    def optimize(self, problem: JointProblem) -> JointSolution:
        raise NotImplementedError

    @staticmethod
    def _invoke(method: Any, item: Any, nodes: Sequence[NodeCandidate],
                edges: Sequence[EdgeCandidate], default: Any) -> Any:
        if method is None:
            return default
        # Accommodate simple constraints and state-aware constraints without a
        # required inheritance hierarchy.
        for args in ((item, nodes, edges), (item, nodes), (item,)):
            try:
                return method(*args)
            except TypeError:
                continue
        return method(item, nodes, edges)

    def allow_node(self, problem: JointProblem, node: NodeCandidate,
                   nodes: Sequence[NodeCandidate], edges: Sequence[EdgeCandidate]) -> bool:
        return all(bool(self._invoke(getattr(c, "allow_node", None), node, nodes, edges, True))
                   for c in problem.constraints)

    @staticmethod
    def _resolve_edge(problem: JointProblem, edge: EdgeCandidate) -> Any:
        from types import SimpleNamespace
        node_by_id = problem.node_by_id
        return SimpleNamespace(
            relation_type=edge.relation_type, type=edge.relation_type,
            head=node_by_id.get(edge.head, edge.head),
            tail=node_by_id.get(edge.tail, edge.tail),
            slot=edge.slot, candidate_id=edge.candidate_id)

    def allow_edge(self, problem: JointProblem, edge: EdgeCandidate,
                   nodes: Sequence[NodeCandidate], edges: Sequence[EdgeCandidate]) -> bool:
        candidate = self._resolve_edge(problem, edge)
        accepted = tuple(self._resolve_edge(problem, value) for value in edges)
        return all(bool(self._invoke(getattr(c, "allow_edge", None), candidate, nodes, accepted, True))
                   for c in problem.constraints)

    def validate_solution(self, problem: JointProblem, solution: JointSolution) -> bool:
        """Recheck every hard constraint against the *final* assignment.

        Incremental construction guarantees feasibility of the picked nodes and
        edges, but :meth:`solution` may inject derived companion edges for
        symmetric/inverse relations that were never screened. This replays the
        incremental checks over the full assignment and then runs whole-result
        ``validate`` hooks (e.g. symmetric/inverse completeness), so a returned
        solution is never allowed to silently violate a declared constraint.
        """
        accepted_nodes: List[NodeCandidate] = []
        for node in solution.nodes:
            if not self.allow_node(problem, node, accepted_nodes + [node], solution.edges):
                return False
            accepted_nodes.append(node)
        accepted_edges: List[EdgeCandidate] = []
        for edge in solution.edges:
            if not self.allow_edge(problem, edge, list(solution.nodes), accepted_edges):
                return False
            accepted_edges.append(edge)
        resolved_relations = tuple(self._resolve_edge(problem, edge) for edge in solution.edges)
        for constraint in problem.constraints:
            validate = getattr(constraint, "validate", None)
            if validate is None:
                continue
            try:
                ok = validate(resolved_relations, list(solution.nodes))
            except TypeError:
                ok = validate(resolved_relations)
            if not ok:
                return False
        return True

    def edge_penalty(self, problem: JointProblem, edge: EdgeCandidate,
                     nodes: Sequence[NodeCandidate], edges: Sequence[EdgeCandidate]) -> float:
        return sum(float(self._invoke(getattr(c, "penalty_edge", None), edge, nodes, edges, 0.0))
                   for c in problem.constraints)

    @staticmethod
    def edge_conflicts(edge: EdgeCandidate, used: Iterable[Hashable]) -> bool:
        used_set = set(used)
        if any(key in used_set for key in edge.exclusion_keys):
            return True
        choice = edge.count_choice
        if choice is None:
            return False
        group, alternative = choice
        return any(
            isinstance(key, tuple) and len(key) == 3 and key[0] == "count-choice"
            and key[1] == group and key[2] != alternative
            for key in used_set
        )

    @staticmethod
    def edge_usage(edge: EdgeCandidate) -> FrozenSet[Hashable]:
        keys = set(edge.exclusion_keys)
        if edge.count_choice is not None:
            keys.add(("count-choice",) + edge.count_choice)
        return frozenset(keys)

    @staticmethod
    def solution(problem: JointProblem, node_ids: Iterable[Hashable],
                 edges: Iterable[EdgeCandidate], score: float) -> JointSolution:
        selected = set(node_ids)
        nodes = tuple(node for node in problem.nodes if node.candidate_id in selected)
        chosen = list(edges)
        keys = {(edge.relation_type, edge.head, edge.tail) for edge in chosen}
        companions = []
        for constraint in problem.constraints:
            if isinstance(constraint, SymmetricRelation):
                pairs = [(constraint.relation, edge.tail, edge.head, edge)
                         for edge in chosen if edge.relation_type == constraint.relation]
            elif isinstance(constraint, InverseRelation):
                pairs = []
                for edge in chosen:
                    if edge.relation_type == constraint.relation:
                        pairs.append((constraint.inverse, edge.tail, edge.head, edge))
                    elif edge.relation_type == constraint.inverse:
                        pairs.append((constraint.relation, edge.tail, edge.head, edge))
            else:
                continue
            for relation, head, tail, source in pairs:
                key = (relation, head, tail)
                if key in keys:
                    continue
                derived = EdgeCandidate(relation, head, tail, 0.0,
                    head_probability=source.tail_probability,
                    tail_probability=source.head_probability,
                    head_entity_probability=source.tail_entity_probability,
                    tail_entity_probability=source.head_entity_probability,
                    count_probability=source.count_probability, derived=True,
                    candidate_id=("derived", relation, head, tail))
                companions.append(derived); keys.add(key)
        chosen_edges = tuple(sorted(chosen + companions,
            key=lambda e: (e.relation_type, str(e.head), str(e.tail), e.derived)))
        return JointSolution(nodes, chosen_edges, float(score))
