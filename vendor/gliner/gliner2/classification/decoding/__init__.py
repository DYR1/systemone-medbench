"""Decoder dispatch and the infeasibility ladder.

``decode`` selects a decoder, runs it, and if the result is infeasible walks the
ladder: ``relax`` (widen retention and retry) -> ``min_violations`` -> ``raise``.
The ladder order and the "never fabricate a feasible-looking violating answer"
rule are the whole point of this module.
"""
from __future__ import annotations

from ..errors import InfeasibleError
from .base import (
    DecodeProblem,
    SearchAssignment,
    Solution,
    build_problem,
)
from .beam import BeamDecoder
from .exact import ExactDecoder, MinViolationsDecoder, _BudgetExceeded
from .independent import IndependentDecoder

__all__ = [
    "DecodeProblem", "SearchAssignment", "Solution", "build_problem",
    "BeamDecoder", "ExactDecoder", "MinViolationsDecoder", "IndependentDecoder",
    "decode", "select_decoder",
]


def select_decoder(problem, requested: str) -> str:
    if requested != "auto":
        return requested
    cross_task = any(len(c.references()) > 1 for c in problem.constraints)
    return "exact" if cross_task else "independent"


def _primary(problem, config):
    """Run the chosen decoder; return a *feasible* Solution or None."""
    decoder = select_decoder(problem, config.decoder)
    if decoder == "independent":
        sol = IndependentDecoder().decode(problem)
        return sol if sol.feasible else None
    if decoder == "beam":
        sol = BeamDecoder().decode(problem, beam_size=config.beam_size)
        return sol if (sol is not None and sol.feasible) else None
    # exact (default), with beam as the budget fallback
    try:
        sol = ExactDecoder().decode(problem, budget=config.exact_node_budget)
    except _BudgetExceeded:
        sol = BeamDecoder().decode(problem, beam_size=config.beam_size)
    return sol if (sol is not None and sol.feasible) else None


def decode(problem, config, *, widen=None) -> Solution:
    sol = _primary(problem, config)
    if sol is not None:
        return sol

    mode = config.on_infeasible
    working = problem
    if mode == "relax" and widen is not None:
        relaxed = widen()
        recovered = _primary(relaxed, config)
        if recovered is not None:
            return recovered
        working = relaxed  # relax failed; min_violations on the widened problem

    if mode in ("relax", "min_violations"):
        return MinViolationsDecoder().decode(working, budget=config.exact_node_budget)

    # raise: diagnose the minimal violation set for the payload
    diagnosis = MinViolationsDecoder().decode(working, budget=config.exact_node_budget)
    raise InfeasibleError(
        "no assignment satisfies the classification constraints",
        violations=diagnosis.violations,
    )
