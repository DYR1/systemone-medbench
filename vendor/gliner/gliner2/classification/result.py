"""Immutable result objects and the builder that assembles them.

Unlike ``joint_ie/result.py`` (whose ``JointResult`` is a mutable dataclass with
public list fields), these are genuinely frozen: ``tasks`` and the per-label
maps are ``MappingProxyType``, so a returned result cannot be silently mutated
by a caller and re-read as if it were the model's output.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

from ..joint_ie.result import _geometric_mean
from .errors import SchemaError


@dataclass(frozen=True)
class Violation:
    constraint: object
    tasks: tuple
    weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))

    def __str__(self) -> str:
        tasks = ", ".join(self.tasks)
        return f"violated {self.constraint} (tasks: {tasks}; weight {self.weight})"


@dataclass(frozen=True)
class TaskResult:
    task: str
    labels: tuple
    probabilities: MappingProxyType
    utilities: MappingProxyType
    confidence: Optional[float]
    exclusive: bool
    ordered: bool
    level: Optional[int]

    def __post_init__(self):
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "probabilities",
                           MappingProxyType(dict(self.probabilities)))
        object.__setattr__(self, "utilities",
                           MappingProxyType(dict(self.utilities)))

    @property
    def label(self) -> Optional[str]:
        if not self.exclusive:
            raise SchemaError(
                f"task {self.task!r} is not single-label; use .labels, not .label"
            )
        return self.labels[0] if self.labels else None


@dataclass(frozen=True)
class ClassificationResult:
    text: str
    tasks: MappingProxyType
    feasible: bool
    violations: tuple
    objective: float
    decoder: str
    exact: bool
    include_confidence: bool = True

    def __post_init__(self):
        object.__setattr__(self, "tasks", MappingProxyType(dict(self.tasks)))
        object.__setattr__(self, "violations", tuple(self.violations))

    def __getitem__(self, task: str) -> TaskResult:
        try:
            return self.tasks[task]
        except KeyError:
            raise SchemaError(f"unknown task {task!r}") from None

    def value(self, task: str):
        tr = self[task]
        return tr.label if tr.exclusive else tr.labels

    def selected(self, task: str) -> tuple:
        return self[task].labels

    def confidence(self, task: str) -> Optional[float]:
        return self[task].confidence

    def utility(self, task: str, label: str) -> float:
        return self[task].utilities[label]

    def probabilities(self, task: str) -> MappingProxyType:
        return self[task].probabilities

    def to_dict(self, *, include_confidence: Optional[bool] = None) -> dict:
        include = self.include_confidence if include_confidence is None else include_confidence
        out: dict = {}
        for name, tr in self.tasks.items():
            value = tr.label if tr.exclusive else list(tr.labels)
            if include:
                out[name] = {
                    "value": value,
                    "confidence": tr.confidence,
                    "probabilities": dict(tr.probabilities),
                }
            else:
                out[name] = value
        if include or not self.feasible:
            out["_meta"] = {
                "feasible": self.feasible,
                "decoder": self.decoder,
                "exact": self.exact,
                "objective": self.objective,
                "violations": [str(v) for v in self.violations],
            }
        return out


class ResultBuilder:
    def build(self, compiled, scores, solution, *, active_order=None,
              include_confidence=True) -> ClassificationResult:
        order = active_order or [t for t in compiled.task_order
                                 if t in solution.assignments]
        task_results = {}
        for task in order:
            spec = compiled.task(task)
            selected = solution.assignments[task].labels
            labels = tuple(l for l in spec.label_names if l in selected)
            probs = {l: scores.probability(task, l) for l in spec.label_names}
            utils = {l: scores.utility(task, l) for l in spec.label_names}
            level = None
            if spec.ordered and labels:
                level = spec.label_names.index(labels[0])
            confidence = self._confidence(spec, labels, probs) if include_confidence else None
            task_results[task] = TaskResult(
                task=task, labels=labels,
                probabilities=probs, utilities=utils,
                confidence=confidence, exclusive=spec.is_exclusive,
                ordered=spec.ordered, level=level,
            )

        violations = tuple(
            Violation(c, tuple(sorted(c.references())))
            for c in solution.violations
        )
        return ClassificationResult(
            text=scores.text,
            tasks=task_results,
            feasible=solution.feasible,
            violations=violations,
            objective=float(solution.score),
            decoder=solution.decoder,
            exact=solution.exact,
            include_confidence=include_confidence,
        )

    @staticmethod
    def _confidence(spec, labels, probs) -> Optional[float]:
        if not labels:
            return 1.0 if spec.default is not None else None
        if spec.is_exclusive:
            return probs[labels[0]]
        selected = set(labels)
        components = [probs[l] if l in selected else 1.0 - probs[l]
                      for l in spec.label_names]
        return _geometric_mean(components)
