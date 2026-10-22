"""Deterministic beam search for joint candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import FrozenSet, Hashable, List, Tuple

from .base import BaseOptimizer, JointSolution
from .greedy import GreedyOptimizer
from ..candidates import EdgeCandidate, JointProblem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _State:
    node_ids: FrozenSet[Hashable] = frozenset()
    edges: Tuple[EdgeCandidate, ...] = ()
    used: FrozenSet[Hashable] = frozenset()
    score: float = 0.0


class BeamOptimizer(BaseOptimizer):
    def __init__(self, beam_width: int = 16) -> None:
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        self.beam_width = beam_width

    @staticmethod
    def _signature(state: _State):
        return (tuple(sorted(map(str, state.node_ids))),
                tuple(str(edge.candidate_id) for edge in state.edges))

    def _finish_nodes(self, problem: JointProblem, state: _State) -> _State:
        node_by_id = problem.node_by_id
        selected = set(state.node_ids)
        score = state.score
        for node in sorted(problem.nodes, key=lambda n: (-n.score, n.entity_type, n.start, n.end)):
            if node.candidate_id in selected or node.score <= 0.0:
                continue
            nodes = [node_by_id[value] for value in selected]
            if self.allow_node(problem, node, nodes + [node], state.edges):
                selected.add(node.candidate_id)
                score += node.score
        return _State(frozenset(selected), state.edges, state.used, score)

    def optimize(self, problem: JointProblem) -> JointSolution:
        node_by_id = problem.node_by_id
        ordered = sorted(problem.edges, key=lambda e: (
            -(e.score + node_by_id[e.head].score + node_by_id[e.tail].score),
            e.relation_type, str(e.hypothesis), str(e.slot), str(e.head), str(e.tail),
        ))
        beam = [_State()]
        for edge in ordered:
            expanded = list(beam)  # explicit skip alternative
            for state in beam:
                if self.edge_conflicts(edge, state.used):
                    continue
                new_ids = [value for value in (edge.head, edge.tail) if value not in state.node_ids]
                added_nodes = [node_by_id[value] for value in new_ids]
                current_nodes = [node_by_id[value] for value in state.node_ids]
                proposed_nodes = current_nodes + added_nodes
                if not all(self.allow_node(problem, node, proposed_nodes, state.edges)
                           for node in added_nodes):
                    continue
                if not self.allow_edge(problem, edge, proposed_nodes, state.edges):
                    continue
                gain = edge.score + sum(node.score for node in added_nodes)
                gain -= self.edge_penalty(problem, edge, proposed_nodes, state.edges)
                if gain < 0.0:
                    continue
                expanded.append(_State(
                    state.node_ids.union(new_ids), state.edges + (edge,),
                    state.used.union(self.edge_usage(edge)), state.score + gain,
                ))
            # Collapse equivalent selections before a stable score/signature cut.
            unique = {}
            for state in expanded:
                key = (state.node_ids, frozenset(e.candidate_id for e in state.edges), state.used)
                old = unique.get(key)
                if old is None or state.score > old.score:
                    unique[key] = state
            beam = sorted(unique.values(), key=lambda s: (-s.score, self._signature(s)))[:self.beam_width]

        # Materialize every candidate assignment (beam finishes plus the greedy
        # baseline) and keep only those that satisfy the hard constraints after
        # derived companions are injected. Greedy no longer substitutes
        # unconditionally: a higher-scoring but infeasible assignment must not win.
        candidates: List[JointSolution] = [
            self.solution(problem, state.node_ids, state.edges, state.score)
            for state in (self._finish_nodes(problem, s) for s in beam)
        ]
        candidates.append(GreedyOptimizer().optimize(problem))

        feasible = [sol for sol in candidates if self.validate_solution(problem, sol)]
        if feasible:
            return max(feasible, key=self._solution_key)

        # No non-trivial feasible assignment exists. The empty solution is always
        # feasible; return it with an explicit infeasibility signal so callers can
        # distinguish "nothing to extract" from "constraints could not be met".
        logger.warning(
            "joint-IE decoding found no constraint-satisfying assignment among %d "
            "candidates; returning empty solution",
            len(candidates),
        )
        empty = self.solution(problem, frozenset(), (), 0.0)
        return replace(empty, feasible=False)

    def _solution_key(self, solution: JointSolution):
        return (solution.score,
                tuple(sorted(map(str, solution.node_ids))),
                tuple(str(edge.candidate_id) for edge in solution.edges))
