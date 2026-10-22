"""Independent decoder: no cross-task coupling, pick each task's best local.

This is what ``decoder="auto"`` selects for schemas with no cross-task
constraint, which is the common case. Single-task constraints (e.g. a lowered
default rule) are still respected by choosing the highest-utility local that
satisfies them.
"""
from __future__ import annotations

from .base import Solution


class IndependentDecoder:
    name = "independent"

    def decode(self, problem) -> Solution:
        chosen: dict = {}
        for task in problem.task_order:
            touching = problem.constraints_touching(task)
            picked = None
            for local in problem.locals[task]:            # utility-descending
                a = problem.assignment({task: local}, [task])
                if all(c.evaluate(a) is not False for c in touching):
                    picked = local
                    break
            if picked is None:
                picked = problem.locals[task][0]          # infeasible; best effort
            chosen[task] = picked
        score = sum(la.utility for la in chosen.values())
        violations = problem.violations_of(chosen)
        return Solution(assignments=dict(chosen), score=score,
                        violations=violations, exact=True, decoder=self.name)
