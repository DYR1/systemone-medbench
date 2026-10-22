"""Constraint AST, three-valued (Kleene) evaluation, and the declarative DSL.

Hard constraints only. One evaluation method returns Kleene-3 logic:
``True`` = satisfied, ``False`` = violated, ``None`` = undetermined.
``still_satisfiable``/``satisfied`` are derived from ``evaluate`` and never
overridden, so they cannot drift apart.

The ``Assignment`` carries *domains*, not just decisions, which is what makes
pruning work: an ordinal or cardinality node can decide before its task is.

Serialization is written explicitly per node (never inherited): an AST cannot be
round-tripped by ``self.__dict__`` (it holds ``Constraint`` objects) nor rebuilt
by a flat ``cls(**values)`` (tuple fields come back as lists).

This module imports neither ``scoring``, ``decoding`` nor ``torch``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, final, runtime_checkable

from .errors import SchemaError


# ======================================================================
# Assignment protocol + reference implementation
# ======================================================================

@runtime_checkable
class Assignment(Protocol):
    """What a constraint reads. Implemented by DictAssignment and, in Phase 6,
    by the decoder's SearchAssignment over live search state."""

    def is_decided(self, task: str) -> bool: ...          # pragma: no cover
    def selected(self, task: str) -> "frozenset[str]": ...  # pragma: no cover
    def domain(self, task: str) -> "frozenset[str]": ...    # pragma: no cover
    def holds(self, task: str, label: str) -> "Optional[bool]": ...  # pragma: no cover
    def levels(self, task: str) -> "frozenset[int]": ...    # pragma: no cover
    def index(self, task: str, label: str) -> int: ...      # pragma: no cover
    def default(self, task: str) -> Optional[str]: ...      # pragma: no cover


def _task_spec(schema, task):
    """Resolve a TaskSpec from either a builder schema or a compiled schema."""
    getter = getattr(schema, "task_spec", None)
    if getter is not None:
        return getter(task)
    return schema.task(task)


class DictAssignment:
    """Test/reference implementation: selected + decided + domains, no search
    state. Both this and the decoder's SearchAssignment satisfy ``Assignment``.
    """

    def __init__(self, schema, selected, decided, domains=None):
        self._schema = schema
        self._selected = {k: frozenset(v) for k, v in (selected or {}).items()}
        self._decided = frozenset(decided or ())
        self._domains = {k: frozenset(v) for k, v in (domains or {}).items()}

    def is_decided(self, task):
        return task in self._decided

    def selected(self, task):
        return self._selected.get(task, frozenset())

    def domain(self, task):
        if task in self._decided:
            return self._selected.get(task, frozenset())
        if task in self._domains:
            return self._domains[task]
        return frozenset(_task_spec(self._schema, task).label_names)

    def holds(self, task, label):
        if label in self.selected(task):
            return True
        if label in self.domain(task):
            return False if self.is_decided(task) else None
        return False

    def levels(self, task):
        spec = _task_spec(self._schema, task)
        index = {name: i for i, name in enumerate(spec.label_names)}
        return frozenset(index[l] for l in self.domain(task) if l in index)

    def index(self, task, label):
        return _task_spec(self._schema, task).label_names.index(label)

    def default(self, task):
        return _task_spec(self._schema, task).default


def _selectable(a: Assignment, task: str) -> "frozenset[str]":
    """Labels that are or could still become selected."""
    return a.selected(task) | a.domain(task)


# ======================================================================
# Kleene-3 helpers (write once; never inline three-valued logic elsewhere)
# ======================================================================

def k_not(v):
    return None if v is None else (not v)


def k_and(values):
    seen_none = False
    for v in values:
        if v is False:
            return False
        if v is None:
            seen_none = True
    return None if seen_none else True


def k_or(values):
    seen_none = False
    for v in values:
        if v is True:
            return True
        if v is None:
            seen_none = True
    return None if seen_none else False


def k_implies(a, b):
    return k_or([k_not(a), b])


def k_iff(a, b):
    if a is None or b is None:
        return None
    return a == b


# ======================================================================
# Constraint base
# ======================================================================

class Constraint(ABC):
    @abstractmethod
    def evaluate(self, a: Assignment):
        """True = satisfied, False = violated, None = undetermined."""

    @abstractmethod
    def references(self) -> "frozenset[str]":
        """Task names touched; drives subset()/drop() and validation."""

    def label_references(self) -> "frozenset[tuple[str, str]]":
        """(task, label) pairs; drives candidate rescue and bound-label
        partitioning. Empty for pure cardinality/ordinal/set nodes."""
        return frozenset()

    def set_references(self) -> "frozenset[str]":
        """Tasks whose selection *set* (via a set predicate) this node reads.
        Drives full subset enumeration in candidates.py."""
        return frozenset()

    def count_references(self) -> "frozenset[str]":
        """Tasks whose selection *count* (via a cardinality node) this node
        reads. Drives per-count enumeration in candidates.py."""
        return frozenset()

    def coupling_references(self) -> "frozenset[str]":
        """Tasks whose whole selection (a count or a set predicate), not just an
        individual label's truth, this node depends on."""
        return self.set_references() | self.count_references()

    def check_schema(self, schema) -> None:
        """Compile-time check; raise SchemaError. Default: no-op."""

    @final
    def still_satisfiable(self, a: Assignment) -> bool:
        return self.evaluate(a) is not False

    @final
    def satisfied(self, a: Assignment) -> bool:
        return self.evaluate(a) is True

    @abstractmethod
    def to_dict(self) -> dict:
        ...


# ---- leaf nodes --------------------------------------------------------

@dataclass(frozen=True)
class LabelRef(Constraint):
    task: str
    label: str

    def evaluate(self, a):
        return a.holds(self.task, self.label)

    def references(self):
        return frozenset({self.task})

    def label_references(self):
        return frozenset({(self.task, self.label)})

    def check_schema(self, schema):
        spec = _task_spec(schema, self.task)
        if self.label not in spec.label_names:
            raise SchemaError(f"{self.label!r} is not a label of {self.task!r}")

    def to_dict(self):
        return {"type": "LabelRef", "task": self.task, "label": self.label}


@dataclass(frozen=True)
class AnySelected(Constraint):
    task: str

    def evaluate(self, a):
        return k_or([a.holds(self.task, l) for l in _selectable(a, self.task)])

    def references(self):
        return frozenset({self.task})

    def set_references(self):
        return frozenset({self.task})

    def check_schema(self, schema):
        _task_spec(schema, self.task)

    def to_dict(self):
        return {"type": "AnySelected", "task": self.task}


@dataclass(frozen=True)
class AnyOtherSelected(Constraint):
    """>=1 NON-default label selected. Requires a declared default."""
    task: str

    def evaluate(self, a):
        default = a.default(self.task)
        return k_or([a.holds(self.task, l)
                     for l in _selectable(a, self.task) if l != default])

    def references(self):
        return frozenset({self.task})

    def set_references(self):
        return frozenset({self.task})

    def check_schema(self, schema):
        spec = _task_spec(schema, self.task)
        if spec.default is None:
            raise SchemaError(
                f"any_other_selected requires task {self.task!r} to declare a default"
            )

    def to_dict(self):
        return {"type": "AnyOtherSelected", "task": self.task}


@dataclass(frozen=True)
class IsDefault(Constraint):
    """The default label is selected. Requires a declared default."""
    task: str

    def evaluate(self, a):
        default = a.default(self.task)
        if default is None:
            return False
        return a.holds(self.task, default)

    def references(self):
        return frozenset({self.task})

    def set_references(self):
        return frozenset({self.task})

    def check_schema(self, schema):
        spec = _task_spec(schema, self.task)
        if spec.default is None:
            raise SchemaError(
                f"is_default requires task {self.task!r} to declare a default"
            )

    def to_dict(self):
        return {"type": "IsDefault", "task": self.task}


@dataclass(frozen=True)
class Cardinality(Constraint):
    task: str
    minimum: int = 0
    maximum: Optional[int] = None

    def evaluate(self, a):
        sel = a.selected(self.task)
        dom = a.domain(self.task)
        lo = len(sel)
        hi = len(sel | dom)
        mx = self.maximum if self.maximum is not None else hi
        if lo > mx:
            return False
        if hi < self.minimum:
            return False
        if lo >= self.minimum and hi <= mx:
            return True
        return None

    def references(self):
        return frozenset({self.task})

    def count_references(self):
        return frozenset({self.task})

    def check_schema(self, schema):
        spec = _task_spec(schema, self.task)
        n = len(spec.label_names)
        if not 0 <= self.minimum <= n:
            raise SchemaError(
                f"cardinality minimum {self.minimum} outside [0, {n}] for {self.task!r}"
            )
        if self.maximum is not None:
            if not 0 <= self.maximum <= n:
                raise SchemaError(
                    f"cardinality maximum {self.maximum} outside [0, {n}] for {self.task!r}"
                )
            if self.minimum > self.maximum:
                raise SchemaError(
                    f"cardinality minimum {self.minimum} exceeds maximum {self.maximum} "
                    f"for {self.task!r}"
                )

    def to_dict(self):
        return {"type": "Cardinality", "task": self.task,
                "minimum": self.minimum, "maximum": self.maximum}


class _OrdinalNode(Constraint):
    """Shared check_schema for ordinal nodes (task exists, ordered, level valid)."""
    task: str
    level: str

    def references(self):
        return frozenset({self.task})

    def check_schema(self, schema):
        spec = _task_spec(schema, self.task)
        if not spec.ordered:
            raise SchemaError(
                f"ordinal op requires an ordered task; {self.task!r} is unordered"
            )
        if self.level not in spec.label_names:
            raise SchemaError(f"{self.level!r} is not a label of {self.task!r}")


@dataclass(frozen=True)
class MinLevel(_OrdinalNode):
    task: str
    level: str

    def evaluate(self, a):
        floor = a.index(self.task, self.level)
        levels = a.levels(self.task)
        if not levels:
            return False
        if min(levels) >= floor:
            return True
        if max(levels) < floor:
            return False
        return None

    def to_dict(self):
        return {"type": "MinLevel", "task": self.task, "level": self.level}


@dataclass(frozen=True)
class MaxLevel(_OrdinalNode):
    task: str
    level: str

    def evaluate(self, a):
        ceil = a.index(self.task, self.level)
        levels = a.levels(self.task)
        if not levels:
            return False
        if max(levels) <= ceil:
            return True
        if min(levels) > ceil:
            return False
        return None

    def to_dict(self):
        return {"type": "MaxLevel", "task": self.task, "level": self.level}


@dataclass(frozen=True)
class AtLevel(_OrdinalNode):
    task: str
    level: str

    def evaluate(self, a):
        target = a.index(self.task, self.level)
        levels = a.levels(self.task)
        if not levels:
            return False
        if levels == {target}:
            return True
        if target not in levels:
            return False
        return None

    def to_dict(self):
        return {"type": "AtLevel", "task": self.task, "level": self.level}


# ---- boolean nodes -----------------------------------------------------

@dataclass(frozen=True)
class Not(Constraint):
    child: Constraint

    def evaluate(self, a):
        return k_not(self.child.evaluate(a))

    def references(self):
        return self.child.references()

    def label_references(self):
        return self.child.label_references()

    def set_references(self):
        return self.child.set_references()

    def count_references(self):
        return self.child.count_references()

    def check_schema(self, schema):
        self.child.check_schema(schema)

    def to_dict(self):
        return {"type": "Not", "child": self.child.to_dict()}


class _NaryNode(Constraint):
    """Base for variadic boolean nodes storing ``children`` as a tuple."""
    children: tuple

    def references(self):
        out = frozenset()
        for c in self.children:
            out |= c.references()
        return out

    def label_references(self):
        out = frozenset()
        for c in self.children:
            out |= c.label_references()
        return out

    def set_references(self):
        out = frozenset()
        for c in self.children:
            out |= c.set_references()
        return out

    def count_references(self):
        out = frozenset()
        for c in self.children:
            out |= c.count_references()
        return out

    def check_schema(self, schema):
        for c in self.children:
            c.check_schema(schema)


@dataclass(frozen=True)
class And(_NaryNode):
    children: tuple

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))

    def evaluate(self, a):
        return k_and([c.evaluate(a) for c in self.children])

    def to_dict(self):
        return {"type": "And", "children": [c.to_dict() for c in self.children]}


@dataclass(frozen=True)
class Or(_NaryNode):
    children: tuple

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))

    def evaluate(self, a):
        return k_or([c.evaluate(a) for c in self.children])

    def to_dict(self):
        return {"type": "Or", "children": [c.to_dict() for c in self.children]}


@dataclass(frozen=True)
class ExactlyOneOf(_NaryNode):
    children: tuple

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))

    def evaluate(self, a):
        vals = [c.evaluate(a) for c in self.children]
        trues = sum(1 for v in vals if v is True)
        nones = sum(1 for v in vals if v is None)
        if trues >= 2:
            return False
        if trues == 1:
            return True if nones == 0 else None
        # trues == 0
        return False if nones == 0 else None

    def to_dict(self):
        return {"type": "ExactlyOneOf",
                "children": [c.to_dict() for c in self.children]}


class _BinaryNode(Constraint):
    left: Constraint
    right: Constraint

    def references(self):
        return self.left.references() | self.right.references()

    def label_references(self):
        return self.left.label_references() | self.right.label_references()

    def set_references(self):
        return self.left.set_references() | self.right.set_references()

    def count_references(self):
        return self.left.count_references() | self.right.count_references()

    def check_schema(self, schema):
        self.left.check_schema(schema)
        self.right.check_schema(schema)


@dataclass(frozen=True)
class Implies(_BinaryNode):
    cond: Constraint
    then: Constraint

    @property
    def left(self):
        return self.cond

    @property
    def right(self):
        return self.then

    def evaluate(self, a):
        return k_implies(self.cond.evaluate(a), self.then.evaluate(a))

    def to_dict(self):
        return {"type": "Implies", "cond": self.cond.to_dict(),
                "then": self.then.to_dict()}


@dataclass(frozen=True)
class Iff(_BinaryNode):
    left: Constraint
    right: Constraint

    def evaluate(self, a):
        return k_iff(self.left.evaluate(a), self.right.evaluate(a))

    def to_dict(self):
        return {"type": "Iff", "left": self.left.to_dict(),
                "right": self.right.to_dict()}


@dataclass(frozen=True)
class Excludes(_BinaryNode):
    left: Constraint
    right: Constraint

    def evaluate(self, a):
        return k_not(k_and([self.left.evaluate(a), self.right.evaluate(a)]))

    def to_dict(self):
        return {"type": "Excludes", "left": self.left.to_dict(),
                "right": self.right.to_dict()}


# ======================================================================
# Serialization
# ======================================================================

_TYPES = {c.__name__: c for c in (
    LabelRef, Not, And, Or, Implies, Iff, Excludes, ExactlyOneOf, Cardinality,
    AtLevel, MinLevel, MaxLevel, IsDefault, AnySelected, AnyOtherSelected,
)}


def constraint_from_dict(data: Mapping) -> Constraint:
    values = dict(data)
    kind = values.pop("type", None)
    try:
        cls = _TYPES[kind]
    except KeyError as exc:
        raise SchemaError(f"unknown constraint type {kind!r}") from exc
    return cls(**{k: _rebuild(v) for k, v in values.items()})


def _rebuild(value):
    if isinstance(value, Mapping) and "type" in value:
        return constraint_from_dict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_rebuild(v) for v in value)
    return value


# ======================================================================
# DSL
# ======================================================================

def _expr(value) -> Constraint:
    """Coerce a ('task', 'label') tuple into a LabelRef; pass Constraints through."""
    if isinstance(value, Constraint):
        return value
    if (isinstance(value, tuple) and len(value) == 2
            and all(isinstance(x, str) for x in value)):
        return LabelRef(value[0], value[1])
    raise SchemaError(
        f"cannot interpret {value!r} as a constraint expression; use a "
        f"('task', 'label') tuple or a DSL constructor"
    )


def _int(value, what) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaError(f"{what} must be a non-negative int")
    return value


def _task_name(value, what) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{what} must be a task name string")
    return value


# boolean
def all_of(*exprs) -> Constraint:
    return And(tuple(_expr(e) for e in exprs))


def any_of(*exprs) -> Constraint:
    return Or(tuple(_expr(e) for e in exprs))


def not_(expr) -> Constraint:
    return Not(_expr(expr))


def implies(cond, then) -> Constraint:
    return Implies(_expr(cond), _expr(then))


def iff(left, right) -> Constraint:
    return Iff(_expr(left), _expr(right))


def excludes(left, right) -> Constraint:
    return Excludes(_expr(left), _expr(right))


def exactly_one_of(*exprs) -> Constraint:
    return ExactlyOneOf(tuple(_expr(e) for e in exprs))


# cardinality — legal on any task
def at_least(task, k) -> Constraint:
    return Cardinality(_task_name(task, "at_least task"), _int(k, "at_least count"), None)


def at_most(task, k) -> Constraint:
    return Cardinality(_task_name(task, "at_most task"), 0, _int(k, "at_most count"))


def exactly(task, k) -> Constraint:
    n = _int(k, "exactly count")
    return Cardinality(_task_name(task, "exactly task"), n, n)


# ordinal — requires ordered=True (enforced at check_schema)
def at_level(task, level) -> Constraint:
    return AtLevel(_task_name(task, "at_level task"), level)


def min_level(task, level) -> Constraint:
    return MinLevel(_task_name(task, "min_level task"), level)


def max_level(task, level) -> Constraint:
    return MaxLevel(_task_name(task, "max_level task"), level)


def between_level(task, lo, hi) -> Constraint:
    name = _task_name(task, "between_level task")
    return And((MinLevel(name, lo), MaxLevel(name, hi)))


# label-set predicates
def any_selected(task) -> Constraint:
    return AnySelected(_task_name(task, "any_selected task"))


def any_other_selected(task) -> Constraint:
    return AnyOtherSelected(_task_name(task, "any_other_selected task"))


def is_default(task) -> Constraint:
    return IsDefault(_task_name(task, "is_default task"))


def label(task, name) -> Constraint:
    return LabelRef(_task_name(task, "label task"), name)
