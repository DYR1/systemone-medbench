"""Declarative schema for multi-task classification.

One task primitive: a label set, cardinality bounds, and an optional order.
``single``/``multi``/``ordinal`` are builder sugar over ``task``.

Every string that reaches this module is injected into the model prompt
verbatim (``processor.py:874-900``) and logit-to-label alignment depends on
marker parsing, so declaration-time validation is unbypassable by any string
input.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .errors import SchemaError

# Marker/structural tokens the processor injects. A label or instruction
# containing any of these would silently corrupt logit-to-label alignment.
_RESERVED = ("[P]", "[L]", "[C]", "[E]", "[R]",
             "[DESCRIPTION]", "[EXAMPLE]", "[OUTPUT]", "(", ")")

_ACTIVATIONS = ("auto", "sigmoid", "softmax")


def _clean(value: str, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{what} must be a non-empty string")
    for token in _RESERVED:
        if token in value:
            raise SchemaError(
                f"{what} may not contain {token!r}: label and prompt strings are "
                f"injected into the model prompt verbatim, and this would corrupt "
                f"logit-to-label alignment"
            )
    return value


def _prob(value: Optional[float], field_name: str) -> None:
    if value is not None and not 0 <= value <= 1:
        raise SchemaError(f"{field_name} must be in [0, 1]")


@dataclass(frozen=True)
class LabelSpec:
    name: str
    description: Optional[str] = None

    def __post_init__(self) -> None:
        _clean(self.name, "label name")
        if self.description is not None:
            _clean(self.description, f"description of label {self.name!r}")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    labels: tuple[LabelSpec, ...]
    min_labels: int = 0
    max_labels: Optional[int] = None          # None = unbounded
    ordered: bool = False                     # declaration order is low -> high
    threshold: float = 0.5                    # decision threshold (utility centring)
    candidate_threshold: Optional[float] = None   # retention floor; None -> config default
    activation: str = "auto"                  # "auto" | "sigmoid" | "softmax"
    temperature: float = 1.0                  # per-task logit calibration
    default: Optional[str] = None
    instruction: Optional[str] = None         # compiles to the processor's `prompt` key
    examples: tuple[tuple[str, str], ...] = ()   # (text, label) few-shot pairs

    def __post_init__(self) -> None:
        _clean(self.name, "task name")
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "examples",
                           tuple(tuple(pair) for pair in self.examples))
        if not self.labels:
            raise SchemaError(f"task {self.name!r} must declare at least one label")
        names = [l.name for l in self.labels]
        if len(set(names)) != len(names):
            raise SchemaError(f"task {self.name!r} has duplicate label names")

        if not isinstance(self.min_labels, int) or self.min_labels < 0:
            raise SchemaError(f"task {self.name!r}: min_labels must be a non-negative int")
        if self.max_labels is not None:
            if not isinstance(self.max_labels, int) or self.max_labels < 0:
                raise SchemaError(f"task {self.name!r}: max_labels must be a non-negative int")
            if self.max_labels > len(self.labels):
                raise SchemaError(
                    f"task {self.name!r}: max_labels ({self.max_labels}) exceeds the "
                    f"number of labels ({len(self.labels)})"
                )

        if self.default is not None:
            if self.default not in names:
                raise SchemaError(
                    f"task {self.name!r}: default {self.default!r} is not one of its labels"
                )
            # A task with a default is never empty: force effective min_labels >= 1.
            if self.min_labels < 1:
                object.__setattr__(self, "min_labels", 1)

        if self.max_labels is not None and self.min_labels > self.max_labels:
            raise SchemaError(
                f"task {self.name!r}: min_labels ({self.min_labels}) exceeds "
                f"max_labels ({self.max_labels})"
            )

        if self.ordered and len(self.labels) < 2:
            raise SchemaError(f"ordered task {self.name!r} requires at least two labels")

        # `threshold` feeds probability_to_logit via center_logit, which returns
        # +-inf at 0/1; an infinite offset is never what anyone meant.
        if not isinstance(self.threshold, (int, float)) or not 0 < self.threshold < 1:
            raise SchemaError(f"task {self.name!r}: threshold must be in (0, 1)")
        _prob(self.candidate_threshold, f"task {self.name!r}: candidate_threshold")

        if self.activation not in _ACTIVATIONS:
            raise SchemaError(
                f"task {self.name!r}: activation must be one of {_ACTIVATIONS}"
            )
        if not isinstance(self.temperature, (int, float)) or self.temperature <= 0:
            raise SchemaError(f"task {self.name!r}: temperature must be positive")

        if self.instruction is not None:
            _clean(self.instruction, f"instruction of task {self.name!r}")

        for pair in self.examples:
            if len(pair) != 2:
                raise SchemaError(
                    f"task {self.name!r}: each example must be a (text, label) pair"
                )
            _, label = pair
            if label not in names:
                raise SchemaError(
                    f"task {self.name!r}: example label {label!r} is not one of its labels "
                    f"(the processor drops such examples silently)"
                )

    @property
    def label_names(self) -> tuple[str, ...]:
        return tuple(l.name for l in self.labels)

    @property
    def is_exclusive(self) -> bool:
        return self.min_labels == 1 and self.max_labels == 1

    def effective_max_labels(self) -> int:
        return len(self.labels) if self.max_labels is None else self.max_labels


def _coerce_labels(labels: Any) -> tuple[LabelSpec, ...]:
    if isinstance(labels, Mapping):
        return tuple(LabelSpec(name, desc) for name, desc in labels.items())
    if isinstance(labels, str):
        raise SchemaError("labels must be a list or a {label: description} mapping, not a str")
    coerced = []
    for item in labels:
        if isinstance(item, LabelSpec):
            coerced.append(item)
        elif isinstance(item, str):
            coerced.append(LabelSpec(item))
        else:
            raise SchemaError(f"invalid label entry {item!r}")
    return tuple(coerced)


def _partition(constraints, active):
    """Split constraints by their task references relative to ``active``.

    ``mixed`` (references straddling the active/inactive boundary) is never
    survivable: a half-applied invariant is worse than no invariant.
    """
    keep, pure_drop, mixed = [], [], []
    for c in constraints:
        refs = frozenset(c.references())
        if refs <= active:
            keep.append(c)
        elif refs & active:
            mixed.append(c)
        else:
            pure_drop.append(c)
    return keep, pure_drop, mixed


class ClassificationSchema:
    """Mutable builder; methods return ``self`` (the JointSchema idiom)."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskSpec] = {}
        self._constraints: list = []

    # ---- introspection -------------------------------------------------

    @property
    def task_specs(self) -> tuple[TaskSpec, ...]:
        return tuple(self._tasks.values())

    @property
    def task_order(self) -> tuple[str, ...]:
        return tuple(self._tasks.keys())

    @property
    def constraints(self) -> tuple:
        return tuple(self._constraints)

    def task_spec(self, name: str) -> TaskSpec:
        try:
            return self._tasks[name]
        except KeyError:
            raise SchemaError(f"unknown task {name!r}") from None

    # ---- constructors --------------------------------------------------

    def task(self, name, labels, *, min_labels=0, max_labels=None, ordered=False,
             threshold=0.5, candidate_threshold=None, activation="auto",
             temperature=1.0, default=None, instruction=None,
             examples=()) -> "ClassificationSchema":
        if name in self._tasks:
            raise SchemaError(f"task {name!r} is already defined")
        spec = TaskSpec(
            name=name,
            labels=_coerce_labels(labels),
            min_labels=min_labels,
            max_labels=max_labels,
            ordered=ordered,
            threshold=threshold,
            candidate_threshold=candidate_threshold,
            activation=activation,
            temperature=temperature,
            default=default,
            instruction=instruction,
            examples=tuple(examples),
        )
        self._tasks[name] = spec
        return self

    def single(self, name, labels, **kw) -> "ClassificationSchema":
        kw.setdefault("min_labels", 1)
        kw.setdefault("max_labels", 1)
        return self.task(name, labels, **kw)

    def multi(self, name, labels, **kw) -> "ClassificationSchema":
        return self.task(name, labels, **kw)

    def ordinal(self, name, labels, **kw) -> "ClassificationSchema":
        kw.setdefault("min_labels", 1)
        kw.setdefault("max_labels", 1)
        kw["ordered"] = True
        return self.task(name, labels, **kw)

    # ---- constraints ---------------------------------------------------

    def constrain(self, *expressions) -> "ClassificationSchema":
        """Append constraints, validated against the tasks declared *so far*.

        Constraints are re-validated at compile. Forward references are a silent
        hole on a mutable builder, so referencing a task that has not been
        declared yet raises here, naming the fix.
        """
        declared = set(self._tasks)
        for expr in expressions:
            if not hasattr(expr, "references"):
                raise SchemaError(
                    f"constraint {expr!r} is not a constraint expression; build one "
                    f"with the constraints DSL"
                )
            missing = frozenset(expr.references()) - declared
            if missing:
                name = sorted(missing)[0]
                raise SchemaError(
                    f"constraint references undeclared task {name!r}; declare task "
                    f"{name!r} before constraining it"
                )
            self._constraints.append(expr)
        return self

    # ---- narrowing -----------------------------------------------------

    def subset(self, *names, keep_constraints="strict") -> "ClassificationSchema":
        """Return a NEW schema with only ``names``.

        WARNING (spec 0d): narrowing the task set changes the encoder input, so
        the scores change. This is a *different measurement*, not a filter. For
        "same scores, fewer reported tasks", use the decode-time ``active=``
        mask instead.
        """
        active = self._resolve_names(names)
        return self._narrow(active, keep_constraints)

    def drop(self, *names, keep_constraints="strict") -> "ClassificationSchema":
        """Return a NEW schema with ``names`` removed (complement of ``subset``).

        Carries the same prompt-coupling warning as ``subset``.
        """
        removed = self._resolve_names(names)
        active = frozenset(self._tasks) - removed
        return self._narrow(active, keep_constraints)

    def _resolve_names(self, names) -> frozenset:
        requested = frozenset(names)
        unknown = requested - set(self._tasks)
        if unknown:
            raise SchemaError(f"unknown task(s): {sorted(unknown)}")
        return requested

    def _narrow(self, active: frozenset, keep_constraints: str) -> "ClassificationSchema":
        if keep_constraints not in ("strict", "prune", "error_on_mixed"):
            raise SchemaError(
                "keep_constraints must be 'strict', 'prune' or 'error_on_mixed'"
            )
        keep, pure_drop, mixed = _partition(self._constraints, active)
        if mixed:
            raise SchemaError(
                "cannot narrow: constraint(s) reference both kept and removed tasks; "
                "a half-applied invariant is never safe"
            )
        if keep_constraints == "strict" and pure_drop:
            raise SchemaError(
                "cannot narrow under keep_constraints='strict': would drop "
                f"{len(pure_drop)} constraint(s); use 'prune' or 'error_on_mixed'"
            )
        if keep_constraints == "prune" and pure_drop:
            warnings.warn(
                f"subset/drop dropped {len(pure_drop)} constraint(s) referencing only "
                f"removed tasks; the prompt changes, so remaining scores are a new "
                f"measurement",
                stacklevel=3,
            )
        out = ClassificationSchema()
        for name, spec in self._tasks.items():
            if name in active:
                out._tasks[name] = spec
        out._constraints = list(keep)
        return out

    # ---- serialization -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": 3,
            "tasks": {name: _task_to_dict(spec) for name, spec in self._tasks.items()},
            "constraints": [c.to_dict() for c in self._constraints],
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping) -> "ClassificationSchema":
        schema = cls()
        for name, spec in data.get("tasks", {}).items():
            schema.task(name, **_task_kwargs_from_dict(spec))
        raw_constraints = data.get("constraints", ())
        if raw_constraints:
            from .constraints import constraint_from_dict
            for raw in raw_constraints:
                schema.constrain(constraint_from_dict(raw))
        return schema

    @classmethod
    def from_json(cls, value: str) -> "ClassificationSchema":
        return cls.from_dict(json.loads(value))

    # ---- equality ------------------------------------------------------

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ClassificationSchema):
            return NotImplemented
        return (self._tasks == other._tasks
                and list(self._constraints) == list(other._constraints))

    def __repr__(self) -> str:
        return (f"ClassificationSchema(tasks={list(self._tasks)}, "
                f"constraints={len(self._constraints)})")


def _task_to_dict(spec: TaskSpec) -> dict:
    labels: Any
    descs = {l.name: l.description for l in spec.labels if l.description is not None}
    if descs:
        labels = {l.name: l.description for l in spec.labels}
    else:
        labels = list(spec.label_names)
    out: dict[str, Any] = {"labels": labels}
    if spec.min_labels:
        out["min_labels"] = spec.min_labels
    if spec.max_labels is not None:
        out["max_labels"] = spec.max_labels
    if spec.ordered:
        out["ordered"] = spec.ordered
    if spec.threshold != 0.5:
        out["threshold"] = spec.threshold
    if spec.candidate_threshold is not None:
        out["candidate_threshold"] = spec.candidate_threshold
    if spec.activation != "auto":
        out["activation"] = spec.activation
    if spec.temperature != 1.0:
        out["temperature"] = spec.temperature
    if spec.default is not None:
        out["default"] = spec.default
    if spec.instruction is not None:
        out["instruction"] = spec.instruction
    if spec.examples:
        out["examples"] = [list(pair) for pair in spec.examples]
    return out


def _task_kwargs_from_dict(spec: Mapping) -> dict:
    kwargs = dict(spec)
    labels = kwargs.pop("labels")
    examples = kwargs.pop("examples", None)
    if examples is not None:
        kwargs["examples"] = tuple(tuple(pair) for pair in examples)
    # `default` forces min_labels>=1 in __post_init__; to_dict may have omitted
    # the forced value, so replaying is idempotent regardless.
    kwargs["labels"] = labels
    return kwargs
