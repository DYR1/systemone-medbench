"""Exceptions for the classification module.

``SchemaError`` subclasses ``ValueError`` deliberately: ``joint_ie/schema.py``
and ``inference/schema.py`` already raise ``ValueError``/``TypeError`` for
schema problems, so callers who catch ``ValueError`` around schema construction
keep working.
"""
from __future__ import annotations

from typing import Iterable


class SchemaError(ValueError):
    """Invalid schema, task, label or constraint. Raised at build/compile time."""


class InfeasibleError(RuntimeError):
    """No assignment satisfies the constraints. Carries the violation payload."""

    def __init__(self, message: str, violations: Iterable = ()):
        super().__init__(message)
        self.violations = tuple(violations)
