"""From letter log-probabilities to answers.

A read-out gives one log-probability (or logit; only differences matter) per option letter. Dividing by the temperature
of the question type and normalising over the listed options gives the option probabilities. After a thought, the
second read-out is re-tempered with the think temperature and mixed with the first one. The answer formats and
confidence formulas are those of the TypeSafe /v1/systemone API.
"""
from __future__ import annotations

import math
from typing import Sequence

from .prompt import Row, score_keys


def probabilities(z: Sequence[float], temperature: float) -> list[float]:
    """softmax(z / temperature) over the listed options."""
    m = max(z)
    e = [math.exp((x - m) / temperature) for x in z]
    s = sum(e)
    return [x / s for x in e]


def retemper(p: Sequence[float], temperature: float, think_temperature: float) -> list[float]:
    """Probabilities read at `temperature`, re-read at `think_temperature`: p ** (temperature / think_temperature),
    normalised."""
    q = [max(v, 1e-300) ** (temperature / think_temperature) for v in p]
    s = sum(q)
    return [v / s for v in q]


def mixture(first: Sequence[float], after: Sequence[float], weight: float) -> list[float]:
    """(1 - weight) * first + weight * after: weight 0 keeps the first read-out, 1 the read-out after the thought."""
    return [(1 - weight) * x + weight * y for x, y in zip(first, after)]


def choice_confidence(p: Sequence[float]) -> float:
    """Confidence of a choice answer: how far the top probability sits above an even split, from 0 (even) to 1 (all on
    one option): (p_max - 1/n) / (1 - 1/n), clamped to [0, 1]."""
    n = len(p)
    return 1.0 if n < 2 else min(1.0, max(0.0, (n * max(p) - 1) / (n - 1)))


def score_confidence(p: Sequence[float]) -> float:
    """Confidence of a score answer: 1 minus the probability-weighted distance in levels from the most likely level,
    relative to the same distance for an even spread measured from the middle level, floored at 0."""
    n = len(p)
    m = max(range(n), key=p.__getitem__)
    spread = sum(v * abs(i - m) for i, v in enumerate(p))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even) if even else 1.0


def option_probabilities(row: Row, p: Sequence[float]) -> dict:
    """Probabilities keyed as in the answer: option ids for choice and yes/no ("true", "false"), "0".."n-1" for
    score levels."""
    if row["type"] == "score":
        return {str(i): v for i, v in enumerate(p)}
    return {o["id"]: v for o, v in zip(row["options"], p)}


def answer(row: Row, p: Sequence[float], spec: dict) -> dict:
    """The /v1/systemone answer of a rendered question with option probabilities p (in option order)."""
    if row["type"] == "noul":
        ids = [o["id"] for o in row["options"]]
        return {"type": "noul", "noul": p[ids.index("true")]}
    if row["type"] == "score":
        levels = spec.get("criteria", spec.get("options")) or []
        levels = [levels[k] for k in score_keys(levels)] if isinstance(levels, dict) else list(levels)
        return {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "confidence": score_confidence(p),
                "legend": {str(i): (levels[i] if i < len(levels) else str(i)) for i in range(len(p))},
                "probabilities": {str(i): v for i, v in enumerate(p)}}
    return choice_answer(dict(zip([o["id"] for o in row["options"]], p)))


def choice_answer(dist: dict, none: float | None = None, abstain: bool = False) -> dict:
    """A choice answer from {option id: probability}; ties go to the option listed first.
    With allow_none, `none` is the probability of the extra option (reported as "none_probability") and `abstain`
    says that it was the most likely one: the choice is then null. Without `none` the answer is the plain one."""
    out = {"type": "choice", "choice": None if abstain else max(dist, key=dist.get),
           "confidence": choice_confidence(list(dist.values())), "probabilities": dist}
    if none is not None:
        out["none_probability"] = none
    return out


def split_none(p: Sequence[float]) -> tuple[list[float], float, bool]:
    """A read-out whose last option is the extra "None of the above" -> (probabilities of the listed options
    renormalised to sum to 1, probability of the extra option before that, whether the extra option is strictly more
    likely than every listed one; a tie goes to the listed option)."""
    real, none = list(p[:-1]), p[-1]
    total = sum(real)
    renormalised = [v / total for v in real] if total > 0 else [1.0 / len(real)] * len(real)
    return renormalised, none, none > max(real)


# ---------------------------------------------------------------- choice lists over 26 options

def knockout_groups(keys: Sequence[str], group: int) -> list[list[str]]:
    """Near-equal consecutive groups of at most `group` options."""
    n = -(-len(keys) // group)
    size, extra = divmod(len(keys), n)
    groups, start = [], 0
    for g in range(n):
        end = start + size + (1 if g < extra else 0)
        groups.append(list(keys[start:end]))
        start = end
    return groups


def knockout_finalists(groups: Sequence[Sequence[str]], group_probs: Sequence[Sequence[float]], keep: int,
                       max_options: int) -> tuple[dict, list[str]]:
    """First-round probabilities {option: p} and the finalists: the `keep` best of each group in group order, or the
    `max_options` best overall when more than that qualify."""
    first, finalists = {}, []
    for g, p in zip(groups, group_probs):
        first.update(zip(g, p))
        finalists += [c for c, _ in sorted(zip(g, p), key=lambda x: -x[1])[:keep]]
    if len(finalists) > max_options:
        finalists = sorted(finalists, key=lambda c: -first[c])[:max_options]
    return first, finalists


def knockout_combine(final: dict, first: dict, residual: float) -> dict:
    """Probabilities over every option: the finalists share 1 - residual as in the final round; the other options
    share `residual` in proportion to their first-round probability."""
    rest = [c for c in first if c not in final]
    mass = sum(first[c] for c in rest) or 1.0
    dist = {c: final[c] * (1 - residual) for c in final}
    dist.update({c: residual * first[c] / mass for c in rest})
    z = sum(dist.values())
    return {c: v / z for c, v in dist.items()}
