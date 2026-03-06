"""Manifest read/write and hash computation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PageEntry:
    """A single page entry in the manifest."""

    id: str
    title: str
    path: str
    source_hash: str
    spec_hash: str
    file_hash: str
    sources: list[str]
    audience: str
    theory_scope: bool


@dataclass
class Manifest:
    """The full manifest state."""

    materializer_version: str
    aggregate_hash: str
    pages: list[PageEntry] = field(default_factory=list)


def compute_hash(content: str) -> str:
    """SHA-256, hex-encoded, truncated to 16 chars."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_file_hash(path: Path) -> str:
    """SHA-256 of file bytes on disk, truncated to 16 chars."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def compute_source_hash(source_paths: list[str], repo_root: Path) -> str:
    """Hash of concatenated source file contents, sorted by path."""
    parts = []
    for sp in sorted(source_paths):
        p = repo_root / sp
        if p.exists():
            parts.append(p.read_text())
        else:
            parts.append(f"__missing__:{sp}")
    return compute_hash("\n".join(parts))


def compute_aggregate_hash(pages: list[PageEntry]) -> str:
    """Hash of all file_hash values sorted by page id."""
    combined = "\n".join(
        p.file_hash for p in sorted(pages, key=lambda x: x.id)
    )
    return compute_hash(combined)


def save_manifest(manifest: Manifest, output_path: Path) -> None:
    """Write manifest to JSON."""
    data = {
        "materializer_version": manifest.materializer_version,
        "aggregate_hash": manifest.aggregate_hash,
        "pages": [
            {
                "id": p.id,
                "title": p.title,
                "path": p.path,
                "source_hash": p.source_hash,
                "spec_hash": p.spec_hash,
                "file_hash": p.file_hash,
                "sources": p.sources,
                "audience": p.audience,
                "theory_scope": p.theory_scope,
            }
            for p in sorted(manifest.pages, key=lambda x: x.id)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def load_manifest(manifest_path: Path) -> Manifest | None:
    """Load manifest from JSON. Returns None if file doesn't exist."""
    if not manifest_path.exists():
        return None

    data = json.loads(manifest_path.read_text())
    pages = [
        PageEntry(
            id=p["id"],
            title=p["title"],
            path=p["path"],
            source_hash=p["source_hash"],
            spec_hash=p["spec_hash"],
            file_hash=p["file_hash"],
            sources=p["sources"],
            audience=p["audience"],
            theory_scope=p["theory_scope"],
        )
        for p in data.get("pages", [])
    ]
    return Manifest(
        materializer_version=data.get("materializer_version", ""),
        aggregate_hash=data.get("aggregate_hash", ""),
        pages=pages,
    )
