"""Architecture-dispatching loader for GLiNER2 extractors.

``AutoExtractor`` resolves the architecture from a checkpoint's config and
returns the matching public model class. Legacy checkpoints without an
``architecture`` field load as ``"span"``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Type

from gliner2.configuration import (
    ExtractorConfig,
    architecture_from_config,
    normalize_architecture,
)
from gliner2.models.loading import (
    HUB_LOAD_OPTIONS,
    checkpoint_file,
    split_load_kwargs,
)


class UnknownArchitectureError(ValueError):
    pass


class ArchitectureMismatchError(ValueError):
    pass


class ArchitectureRegistrationError(ValueError):
    pass


# Backward-compatible private alias; the shared loader owns the option set.
_HUB_LOAD_KEYS = HUB_LOAD_OPTIONS


def _ensure_registered() -> None:
    """Import the engine module so built-in architectures self-register."""
    if set(AutoExtractor._registry) >= {"span", "boundary"}:
        return
    # Importing the engine registers SpanExtractor and BoundaryExtractor.
    import gliner2.inference.engine  # noqa: F401


class AutoExtractor:
    """Registry-based dispatcher over extractor architectures."""

    _registry: ClassVar[Dict[str, Type]] = {}

    # -- registry -----------------------------------------------------------
    @classmethod
    def register(
        cls,
        architecture: str,
        model_class: Type,
        *,
        exist_ok: bool = False,
    ) -> None:
        name = normalize_architecture(architecture)
        if name in cls._registry and not exist_ok:
            if cls._registry[name] is model_class:
                return
            raise ArchitectureRegistrationError(
                f"Architecture {name!r} is already registered to "
                f"{cls._registry[name]!r}. Pass exist_ok=True to override."
            )
        cls._registry[name] = model_class

    @classmethod
    def _resolve_class(cls, architecture: str) -> Type:
        _ensure_registered()
        name = normalize_architecture(architecture)
        if name not in cls._registry:
            raise UnknownArchitectureError(
                f"No model class registered for architecture {name!r}. "
                f"Registered: {sorted(cls._registry)}"
            )
        return cls._registry[name]

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_config(cls, config: ExtractorConfig, **kwargs: Any):
        architecture = architecture_from_config(config)
        model_class = cls._resolve_class(architecture)
        return model_class(config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args: Any,
        architecture: Optional[str] = None,
        config: Optional[ExtractorConfig] = None,
        allow_architecture_override: bool = False,
        **kwargs: Any,
    ):
        model_kwargs, hub_kwargs = split_load_kwargs(
            kwargs, context="AutoExtractor.from_pretrained"
        )

        if config is None:
            config = _load_config(pretrained_model_name_or_path, hub_kwargs)

        saved_architecture = architecture_from_config(config)
        requested_architecture = (
            normalize_architecture(architecture)
            if architecture is not None
            else saved_architecture
        )

        if requested_architecture != saved_architecture and not allow_architecture_override:
            raise ArchitectureMismatchError(
                f"Checkpoint architecture is {saved_architecture!r}, "
                f"but {requested_architecture!r} was requested. "
                "Span and boundary heads are not checkpoint-compatible, and "
                "automatic architecture conversion is not supported. Load the "
                "checkpoint with its saved architecture or initialize and train "
                "a separate boundary checkpoint."
            )

        model_class = cls._resolve_class(requested_architecture)
        return model_class.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            **model_kwargs,
        )


def _load_config(path, hub_kwargs: Dict[str, Any]) -> ExtractorConfig:
    """Load an ``ExtractorConfig`` from a local dir or the Hub."""
    config_file = checkpoint_file(str(path), "config.json", hub_kwargs)
    return ExtractorConfig.from_pretrained(config_file)
