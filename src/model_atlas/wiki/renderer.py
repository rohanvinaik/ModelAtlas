"""Page assembly and frontmatter generation."""

from __future__ import annotations

from pathlib import Path

from .config import PageConfig, WikiConfig
from .extractor import extract_sections
from .manifest import (
    Manifest,
    PageEntry,
    compute_aggregate_hash,
    compute_file_hash,
    compute_source_hash,
    save_manifest,
)


def _render_frontmatter(page: PageConfig, source_hash: str, spec_hash: str,
                        file_hash: str, version: str) -> str:
    """Generate YAML frontmatter for a page."""
    sources_yaml = "\n".join(f"  - {s.path}" for s in page.sources)
    if not sources_yaml:
        sources_yaml = "  []"
    else:
        sources_yaml = "\n" + sources_yaml

    return (
        f"---\n"
        f"generated: true\n"
        f"generated_from: {sources_yaml}\n"
        f"source_hash: {source_hash}\n"
        f"spec_hash: {spec_hash}\n"
        f"file_hash: {file_hash}\n"
        f"materializer_version: \"{version}\"\n"
        f"theory_scope: {'true' if page.theory_scope else 'false'}\n"
        f"audience: {page.audience}\n"
        f"page_id: {page.id}\n"
        f"---\n"
    )


def _render_index(config: WikiConfig) -> str:
    """Generate index page body listing all pages by audience."""
    lines = [f"# {config.pages[0].title if config.pages else 'Wiki'}\n"]
    lines.append("## Pages\n")

    by_audience: dict[str, list[PageConfig]] = {}
    for page in config.pages:
        if page.auto_index:
            continue
        by_audience.setdefault(page.audience, []).append(page)

    for audience in sorted(by_audience):
        lines.append(f"### {audience.title()}\n")
        for page in by_audience[audience]:
            lines.append(f"- [{page.title}]({page.id}.md)")
        lines.append("")

    return "\n".join(lines)


def _render_page_body(page: PageConfig, repo_root: Path) -> str:
    """Render the body content of a page from its sources."""
    if page.auto_index:
        return ""  # handled separately

    parts = [f"# {page.title}\n"]
    for source in page.sources:
        content = extract_sections(source, repo_root)
        if content.strip():
            parts.append(content)

    return "\n\n".join(parts)


def materialize(config: WikiConfig, repo_root: Path, output_dir: Path) -> Manifest:
    """Materialize all wiki pages and write manifest.

    Returns the generated manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[PageEntry] = []

    for page in config.pages:
        source_paths = [s.path for s in page.sources]
        source_hash = compute_source_hash(source_paths, repo_root)
        spec_hash = page.spec_hash()

        # Render body
        if page.auto_index:
            body = _render_index(config)
        else:
            body = _render_page_body(page, repo_root)

        # First pass: render with placeholder file_hash to compute real one
        placeholder = "0" * 16
        content = _render_frontmatter(page, source_hash, spec_hash,
                                      placeholder, config.materializer_version)
        content += "\n" + body

        # Compute file_hash from content with placeholder, then re-render
        # To make this deterministic, we hash the body + metadata (not file_hash itself)
        from .manifest import compute_hash
        stable_parts = f"{source_hash}|{spec_hash}|{config.materializer_version}|{body}"
        file_hash = compute_hash(stable_parts)

        # Final render with real file_hash
        content = _render_frontmatter(page, source_hash, spec_hash,
                                      file_hash, config.materializer_version)
        content += "\n" + body

        # Write page
        page_path = output_dir / f"{page.id}.md"
        page_path.write_text(content)

        # Manifest file_hash = actual bytes hash of the written file.
        # (Frontmatter file_hash is the stable-parts fingerprint for provenance;
        # manifest file_hash is the real bytes hash for drift detection.)
        actual_file_hash = compute_file_hash(page_path)

        entries.append(PageEntry(
            id=page.id,
            title=page.title,
            path=f"{page.id}.md",
            source_hash=source_hash,
            spec_hash=spec_hash,
            file_hash=actual_file_hash,
            sources=source_paths,
            audience=page.audience,
            theory_scope=page.theory_scope,
        ))

    aggregate = compute_aggregate_hash(entries)
    manifest = Manifest(
        materializer_version=config.materializer_version,
        aggregate_hash=aggregate,
        pages=entries,
    )

    save_manifest(manifest, output_dir / "manifest.json")
    return manifest
