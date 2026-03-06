"""Source map parser for wiki.yaml."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceSpec:
    """A single source file reference within a page config."""

    path: str
    sections: list[str] | str = "all"  # list of heading titles, or "all"
    extract: str | None = None  # e.g. "docstrings" for Python files

    def normalized(self) -> dict:
        """Stable dict for hashing."""
        d: dict[str, Any] = {"path": self.path}
        if isinstance(self.sections, list):
            d["sections"] = sorted(self.sections)
        else:
            d["sections"] = self.sections
        if self.extract:
            d["extract"] = self.extract
        return d


@dataclass
class PageConfig:
    """Configuration for a single wiki page."""

    id: str
    title: str
    audience: str
    sources: list[SourceSpec] = field(default_factory=list)
    auto_index: bool = False
    theory_scope: bool = False

    def spec_hash(self) -> str:
        """Hash of this page's normalized config entry."""
        d = {
            "id": self.id,
            "title": self.title,
            "audience": self.audience,
            "auto_index": self.auto_index,
            "sources": [s.normalized() for s in self.sources],
        }
        raw = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class WikiConfig:
    """Parsed wiki.yaml configuration."""

    materializer_version: str
    default_audience: str
    default_theory_scope: bool
    pages: list[PageConfig]
    promotions: list[str] = field(default_factory=list)


def _parse_source(raw: dict) -> SourceSpec:
    sections = raw.get("sections", "all")
    return SourceSpec(
        path=raw["path"],
        sections=sections,
        extract=raw.get("extract"),
    )


def _parse_page(raw: dict, defaults: dict, promotions: list[str]) -> PageConfig:
    sources = [_parse_source(s) for s in raw.get("sources", [])]
    page_id = raw["id"]
    return PageConfig(
        id=page_id,
        title=raw["title"],
        audience=raw.get("audience", defaults.get("audience", "user")),
        sources=sources,
        auto_index=raw.get("auto_index", False),
        theory_scope=page_id in promotions,
    )


def load_config(config_path: Path) -> WikiConfig:
    """Load and parse wiki.yaml."""
    import yaml

    text = config_path.read_text()
    data = yaml.safe_load(text)

    defaults = data.get("defaults", {})
    promotions = [p["page_id"] for p in data.get("promotions", []) if "page_id" in p]

    pages = [_parse_page(p, defaults, promotions) for p in data.get("pages", [])]

    return WikiConfig(
        materializer_version=data.get("materializer_version", "0.0.0"),
        default_audience=defaults.get("audience", "user"),
        default_theory_scope=defaults.get("theory_scope", False),
        pages=pages,
        promotions=promotions,
    )
