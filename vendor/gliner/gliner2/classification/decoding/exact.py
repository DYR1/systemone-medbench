"""Exact decoder: DFS with branch and bound over per-task local assignments.

Most-constrained-first ordering makes coupled constraints prune early. Because
each task's locals are utility-sorted, once the optimistic bound fails for one
local, every remaining local for that task fails too, so we ``break``.

``min_violations`` reuses the same traversal with a lexicographic objective and
no feasibility pruning: one search implementation, two scoring modes.
"""
from __future__ import annotations

import math

from .base import Solution


class _BudgetExceeded(Exception):
    pass


def _order(problem):
    return sorted(
        problem.task_order,
        key=lambda t: (-len(problem.constraints_touching(t)),
                       len(problem.locals[t]), t),
    )


def _suffix_max(order, problem):
    """suffix[i] = best achievable utility for tasks order[i:] independently."""
    suffix = [0.0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        locals_ = problem.locals[order[i]]
        best = max((la.utility for la in locals_), default=0.0)
        suffix[i] = best + suffix[i + 1]
    return suffix


class ExactDecoder:
    name = "exact"

    def decode(self, problem, *, budget: int = 200_000) -> Solution:
        """Return the max-utility feasible Solution, or raise _BudgetExceeded."""
        order = _order(problem)
        suffix = _suffix_max(order, problem)
        best = {"assign": None, "score": -math.inf}
        nodes = {"n": 0}

        def dfs(i, chosen, score):
            nodes["n"] += 1
            if nodes["n"] > budget:
                raise _BudgetExceeded
            if score + suffix[i] <= best["score"]:
                return
            if i == len(order):
                best["assign"] = dict(chosen)
                best["score"] = score
                return
            task = order[i]
            touching = problem.constraints_touching(task)
            for local in problem.locals[task]:                   # utility-descending
                if score + local.utility + suffix[i + 1] <= best["score"]:
                    break                                        # sorted => rest worse
                chosen[task] = local
                a = problem.assignment(chosen, order[:i + 1])
                if all(c.evaluate(a) is not False for c in touching):
                    dfs(i + 1, chosen, score + local.utility)
                del chosen[task]

        dfs(0, {}, 0.0)
        if best["assign"] is None:
            return None  # no feasible assignment within the search
        return Solution(assignments=best["assign"], score=best["score"],
                        violations=(), exact=True, decoder=self.name)


class MinViolationsDecoder:
    """Same DFS, lexicographic objective (fewest violations, then max utility),
    no pruning on ``False``."""
    name = "min_violations"

    def decode(self, problem, *, budget: int = 200_000) -> Solution:
        order = _order(problem)
        best = {"assign": None, "weight": math.inf, "score": -math.inf}
        nodes = {"n": 0}

        def dfs(i, chosen, score):
            nodes["n"] += 1
            if nodes["n"] > budget:
                raise _BudgetExceeded
            if i == len(order):
                violated = problem.violations_of(chosen)
                weight = float(len(violated))
                key = (weight, -score)
                cur = (best["weight"], -best["score"])
                if key < cur:
                    best["assign"] = dict(chosen)
                    best["weight"] = weight
                    best["score"] = score
                return
            task = order[i]
            for local in problem.locals[task]:
                chosen[task] = local
                dfs(i + 1, chosen, score + local.utility)
                del chosen[task]

        exhausted = False
        try:
            dfs(0, {}, 0.0)
        except _BudgetExceeded:
            exhausted = True  # best-so-far is not proven minimal
        assign = best["assign"] or {t: problem.locals[t][0] for t in problem.task_order}
        violated = problem.violations_of(assign)
        score = sum(la.utility for la in assign.values())
        return Solution(assignments=dict(assign), score=score,
                        violations=violated, exact=not exhausted, decoder=self.name)
