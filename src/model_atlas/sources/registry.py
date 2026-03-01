"""Source adapter registry."""

from __future__ import annotations

import logging

from .base import SourceAdapter

logger = logging.getLogger(__name__)

_registry: dict[str, SourceAdapter] = {}


def register_source(adapter: SourceAdapter) -> None:
    """Register a source adapter by its name."""
    _registry[adapter.name] = adapter


def get_source(name: str) -> SourceAdapter:
    """Get a registered source adapter by name. Raises KeyError if not found."""
    return _registry[name]


def list_sources() -> dict[str, SourceAdapter]:
    """Return all registered source adapters."""
    return dict(_registry)


def _auto_register() -> None:
    """Auto-register built-in adapters."""
    from .huggingface import HuggingFaceAdapter

    register_source(HuggingFaceAdapter())

    try:
        from .ollama import OllamaAdapter

        register_source(OllamaAdapter())
    except Exception:
        logger.debug("Ollama adapter not available, skipping registration")


_auto_register()
