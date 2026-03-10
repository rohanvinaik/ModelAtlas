"""Drift detection between source docs and materialized wiki."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import WikiConfig
from .manifest import compute_file_hash, compute_source_hash, load_manifest


@dataclass
class PageDrift:
    """Drift status for a single page."""

    page_id: str
    status: str  # "ok", "stale", "orphaned"
    reasons: list[str] = field(default_factory=list)


@dataclass
class DriftReport:
    """Full drift report across all pages."""

    pages: list[PageDrift] = field(default_factory=list)
    version_mismatch: bool = False

    @property
    def stale_count(self) -> int:
        return sum(1 for p in self.pages if p.status == "stale")

    @property
    def ok_count(self) -> int:
        return sum(1 for p in self.pages if p.status == "ok")

    @property
    def orphaned_count(self) -> int:
        return sum(1 for p in self.pages if p.status == "orphaned")

    @property
    def is_clean(self) -> bool:
        return self.stale_count == 0 and self.orphaned_count == 0

    def format_human(self) -> str:
        """Human-readable drift report."""
        lines = ["Wiki drift report:"]
        for p in self.pages:
            reasons = ", ".join(p.reasons) if p.reasons else ""
            status_str = p.status.upper()
            if reasons:
                lines.append(f"  {p.page_id:<30} {status_str:<10} {reasons}")
            else:
                lines.append(f"  {p.page_id:<30} {status_str}")
        lines.append("")
        lines.append(
            f"{self.stale_count} stale, {self.ok_count} current, "
            f"{self.orphaned_count} orphaned"
        )
        if not self.is_clean:
            lines.append("Run `python -m model_atlas.wiki materialize` to regenerate.")
        return "\n".join(lines)


def check_drift(
    config: WikiConfig,
    repo_root: Path,
    output_dir: Path,
) -> DriftReport:
    """Check for drift between source docs and materialized wiki."""
    manifest = load_manifest(output_dir / "manifest.json")
    report = DriftReport()

    if manifest is None:
        # No manifest — everything is stale
        for page in config.pages:
            report.pages.append(
                PageDrift(
                    page_id=page.id, status="stale", reasons=["no manifest found"]
                )
            )
        return report

    # Check version mismatch
    if manifest.materializer_version != config.materializer_version:
        report.version_mismatch = True
        for page in config.pages:
            report.pages.append(
                PageDrift(
                    page_id=page.id,
                    status="stale",
                    reasons=[
                        f"version mismatch (manifest={manifest.materializer_version}, "
                        f"config={config.materializer_version})"
                    ],
                )
            )
        return report

    # Build lookup from manifest
    manifest_by_id = {p.id: p for p in manifest.pages}
    seen_ids: set[str] = set()

    for page in config.pages:
        seen_ids.add(page.id)
        entry = manifest_by_id.get(page.id)

        if entry is None:
            report.pages.append(
                PageDrift(
                    page_id=page.id,
                    status="stale",
                    reasons=["page not in manifest"],
                )
            )
            continue

        reasons: list[str] = []

        # Check source hash
        source_paths = [s.path for s in page.sources]
        current_source_hash = compute_source_hash(source_paths, repo_root)
        if current_source_hash != entry.source_hash:
            changed = [s.path for s in page.sources if (repo_root / s.path).exists()]
            reasons.append(f"source changed ({', '.join(changed)})")

        # Check spec hash
        current_spec_hash = page.spec_hash()
        if current_spec_hash != entry.spec_hash:
            reasons.append("page config changed")

        # Check file on disk
        page_path = output_dir / f"{page.id}.md"
        if not page_path.exists():
            reasons.append("page file missing")
        else:
            disk_hash = compute_file_hash(page_path)
            if disk_hash != entry.file_hash:
                reasons.append("page file modified on disk")

        status = "stale" if reasons else "ok"
        report.pages.append(
            PageDrift(
                page_id=page.id,
                status=status,
                reasons=reasons,
            )
        )

    # Check for orphaned pages (in manifest but not in config)
    for entry in manifest.pages:
        if entry.id not in seen_ids:
            report.pages.append(
                PageDrift(
                    page_id=entry.id,
                    status="orphaned",
                    reasons=["page in manifest but not in config"],
                )
            )

    return report
