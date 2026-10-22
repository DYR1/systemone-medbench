"""Multi-task classification with hard cross-task constraints (v3 API).

Public surface (9 names). The constraint DSL is the ``constraints`` submodule::

    from gliner2.classification import Classifier, ClassificationSchema
    from gliner2.classification import constraints as C

    schema = (ClassificationSchema()
              .single("intent", ["read", "write", "delete"])
              .multi("effects", ["read_only", "create", "modify", "delete"])
              .constrain(C.implies(("intent", "delete"), ("effects", "delete"))))
    clf = Classifier.from_pretrained("fastino/gliner2.5-base-v1")
    result = clf.classify("rm -rf /", schema)
"""
from __future__ import annotations

from . import constraints
from .compiler import compile_schema
from .engine import ClassificationConfig, Classifier
from .errors import InfeasibleError, SchemaError
from .result import ClassificationResult
from .schema import ClassificationSchema
from .scoring import ClassificationScores

__all__ = [
    "Classifier",
    "ClassificationSchema",
    "ClassificationConfig",
    "ClassificationResult",
    "ClassificationScores",
    "SchemaError",
    "InfeasibleError",
    "compile_schema",
    "constraints",
]
