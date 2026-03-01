"""Base protocol for source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..extraction.deterministic import ModelInput


@dataclass
class SourceSearchResult:
    """Source-agnostic search result."""

    model_id: str
    author: str = ""
    source: str = ""
    display_name: str = ""
    description: str = ""
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    last_modified: str | None = None
    raw: dict = field(default_factory=dict)


class SourceAdapter(ABC):
    """Abstract base class for model source adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this source (e.g. 'huggingface', 'ollama')."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: dict | None = None,
    ) -> list[SourceSearchResult]:
        """Search for models matching the query."""

    @abstractmethod
    def get_detail(self, model_id: str) -> ModelInput:
        """Get full model details as a ModelInput for extraction."""

    def fetch_config(self, model_id: str) -> dict | None:
        """Fetch model config.json. Optional — returns None by default."""
        return None
