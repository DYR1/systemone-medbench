"""
Compatibility shim for the span (legacy) extractor model.

The concrete implementation now lives in
:mod:`gliner2.models.span.model`. This module re-exports the public names so
existing imports keep working unchanged::

    from gliner2.model import Extractor, SpanExtractorModel, ExtractorConfig

State-dict keys, numerical behavior, and the ``Extractor`` public name are
preserved; only the physical location of the class moved.
"""

from gliner2.configuration import ExtractorConfig
from gliner2.models.span.model import Extractor, SpanExtractorModel

__all__ = ["Extractor", "SpanExtractorModel", "ExtractorConfig"]
