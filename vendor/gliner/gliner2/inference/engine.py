"""
GLiNER2 - Advanced Information Extraction Engine

Defines the public model classes for both architectures:

* ``SpanExtractor``    — span (legacy) architecture + shared runtime.
* ``BoundaryExtractor`` — boundary architecture + shared runtime.
* ``GLiNER2``          — backward-compatible span class (``GLiNER2 = SpanExtractor``
  subclass) preserving the historical public name and aliases.

Both public classes mix :class:`ExtractorRuntimeMixin` so they expose an
identical extraction API. Architecture-specific behavior lives in the model
core (candidate production), not in the shared runtime.

Example:
    >>> from gliner2 import GLiNER2
    >>> extractor = GLiNER2.from_pretrained("model-repo")
    >>> results = extractor.extract_entities("Apple released iPhone 15.", ["company", "product"])
"""

from __future__ import annotations

# Back-compat re-exports: existing code that does
#   ``from gliner2.inference.engine import Schema, RegexValidator``
# continues to work unchanged.
from gliner2.inference.schema import (  # noqa: F401 - compatibility exports
    AttributeGroup, RegexValidator, StructureBuilder, Schema
)

from gliner2.auto import AutoExtractor
from gliner2.inference.runtime import ExtractorRuntimeMixin
from gliner2.models.span.model import SpanExtractorModel


class SpanExtractor(ExtractorRuntimeMixin, SpanExtractorModel):
    """Span (legacy) architecture with the shared public extraction runtime."""

    architecture = "span"


class GLiNER2(SpanExtractor):
    """Backward-compatible span architecture class.

    Retains the historical public name. ``GLiNER2.from_pretrained()`` remains
    span-only; use ``AutoExtractor.from_pretrained()`` for architecture
    dispatch.
    """


AutoExtractor.register("span", SpanExtractor, exist_ok=True)

# Boundary is a first-class architecture. Import failures must remain visible:
# suppressing them here makes a broken installation look like an unsupported
# checkpoint later, discarding the actionable root cause.
from gliner2.models.boundary.engine import BoundaryExtractor  # noqa: E402

AutoExtractor.register("boundary", BoundaryExtractor, exist_ok=True)


# Aliases
BuilderExtractor = GLiNER2
SchemaBuilder = Schema
JsonStructBuilder = StructureBuilder
