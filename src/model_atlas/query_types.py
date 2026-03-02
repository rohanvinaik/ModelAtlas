"""Data types for the navigational query engine.

Separated from query.py for better cohesion — types/schemas in one module,
query logic in another. Other modules can import these without pulling in
the full query engine and its dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BankConstraint:
    """A constraint on a single semantic bank."""

    bank: str
    direction: int | None = None  # +1 or -1, None = any
    target_position: int | None = None  # specific signed position to be near
    min_signed: int | None = None
    max_signed: int | None = None
    weight: float = 1.0


@dataclass
class ParsedQuery:
    """Structured representation of a parsed natural language query."""

    bank_constraints: list[BankConstraint]
    anchor_targets: list[str]  # desired anchor labels
    seed_model_ids: list[str]  # "models like X" seeds
    direction_vectors: dict[str, int]  # bank -> desired direction (+1/-1)
    raw_tokens: list[str]  # for fuzzy fallback


@dataclass
class SearchResult:
    """A model returned from a navigational search."""

    model_id: str
    score: float
    bank_score: float = 0.0
    anchor_score: float = 0.0
    spread_score: float = 0.0
    fuzzy_score: float = 0.0
    positions: dict[str, dict] = field(default_factory=dict)
    anchor_labels: list[str] = field(default_factory=list)
    vibe_summary: str = ""
    author: str = ""


@dataclass
class ComparisonResult:
    """Result of comparing two or more models."""

    models: list[str]
    shared_anchors: list[str]
    per_model_unique: dict[str, list[str]]
    jaccard_similarity: float
    bank_deltas: dict[str, dict]


@dataclass
class StructuredQuery:
    """Structured input for navigate_models — the calling LLM fills this in."""

    # Per-bank direction: -1 (toward negative), 0 (near zero), +1 (toward positive)
    architecture: int | None = None
    capability: int | None = None
    efficiency: int | None = None
    compatibility: int | None = None
    lineage: int | None = None
    domain: int | None = None
    quality: int | None = None
    training: int | None = None

    # Anchor targeting
    require_anchors: list[str] = field(default_factory=list)
    prefer_anchors: list[str] = field(default_factory=list)
    avoid_anchors: list[str] = field(default_factory=list)

    # Seed model for similarity
    similar_to: str | None = None

    # Result control
    limit: int = 20

    def bank_directions(self) -> dict[str, int]:
        """Return {bank_name: direction} for all specified banks."""
        mapping = {
            "ARCHITECTURE": self.architecture,
            "CAPABILITY": self.capability,
            "EFFICIENCY": self.efficiency,
            "COMPATIBILITY": self.compatibility,
            "LINEAGE": self.lineage,
            "DOMAIN": self.domain,
            "QUALITY": self.quality,
            "TRAINING": self.training,
        }
        return {k: v for k, v in mapping.items() if v is not None}


@dataclass
class NavigationResult:
    """A model returned from navigate_models."""

    model_id: str
    score: float
    bank_alignment: float = 0.0
    anchor_relevance: float = 0.0
    seed_similarity: float = 0.0
    positions: dict[str, dict] = field(default_factory=dict)
    anchor_labels: list[str] = field(default_factory=list)
    vibe_summary: str = ""
    author: str = ""
