"""Beam decoder: bounded fallback when exact search exceeds its node budget.

Crucially NOT ``joint_ie/optimizers/beam.py``, whose ``optimize`` returns a
greedy-max assignment when the beam empties (``beam.py:86-87``) - for hard
classification constraints that silently emits a *constraint-violating* answer.
Here, an empty beam means "beam could not find a feasible assignment" and we
report exactly that (the caller then falls to min_violations). We never return
a violating assignment as if it were feasible.
"""
from __future__ import annotations

from typing import Optional

from .base import Solution


def _signature(chosen, order):
    return tuple((t, tuple(sorted(chosen[t].labels))) for t in order if t in chosen)


class BeamDecoder:
    name = "beam"

    def decode(self, problem, *, beam_size: int = 16) -> Optional[Solution]:
        """Return the best feasible Solution in the beam, or None at a dead end."""
        order = sorted(
            problem.task_order,
            key=lambda t: (-len(problem.constraints_touching(t)),
                           len(problem.locals[t]), t),
        )
        beams = [(0.0, {})]
        for i, task in enumerate(order):
            touching = problem.constraints_touching(task)
            expanded = []
            seen = set()
            for score, chosen in beams:
                for local in problem.locals[task]:
                    nxt = dict(chosen)
                    nxt[task] = local
                    a = problem.assignment(nxt, order[:i + 1])
                    if all(c.evaluate(a) is not False for c in touching):
                        expanded.append((score + local.utility, nxt))
            expanded.sort(key=lambda item: (-item[0], _signature(item[1], order)))
            beams = []
            for score, chosen in expanded:
                sig = _signature(chosen, order)
                if sig in seen:
                    continue
                seen.add(sig)
                beams.append((score, chosen))
                if len(beams) >= beam_size:
                    break
            if not beams:
                # No feasible extension survived; report infeasibility rather
                # than fabricate a greedy-max (constraint-violating) answer.
                # An empty Solution would read as feasible (no violations) and
                # skip the on_infeasible ladder, so return None like ExactDecoder.
                return None

        score, chosen = beams[0]
        violations = problem.violations_of(chosen)
        return Solution(assignments=dict(chosen), score=score,
                        violations=violations, exact=False, decoder=self.name)
