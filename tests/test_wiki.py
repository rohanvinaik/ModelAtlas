"""Tests for the wiki materializer — determinism, drift, provenance, round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_atlas.wiki.config import SourceSpec, WikiConfig, load_config
from model_atlas.wiki.drift import check_drift
from model_atlas.wiki.extractor import extract_sections
from model_atlas.wiki.manifest import (
    Manifest,
    PageEntry,
    compute_file_hash,
    compute_source_hash,
    load_manifest,
    save_manifest,
)
from model_atlas.wiki.renderer import materialize


@pytest.fixture
def wiki_tree(tmp_path: Path):
    """Create a minimal wiki project tree for testing."""
    # Source doc
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "design.md").write_text(
        "# Overview\n\nThis is the overview.\n\n"
        "# Architecture\n\nArch details here.\n\n"
        "# API\n\nAPI docs.\n"
    )
    (tmp_path / "README.md").write_text(
        "# Project\n\nA short readme.\n\n"
        "## Install\n\nRun pip install.\n"
    )

    # wiki.yaml
    config_text = """\
materializer_version: "0.1.0"
defaults:
  theory_scope: false
  audience: user
pages:
  - id: home
    title: "Home"
    audience: user
    sources: []
    auto_index: true
  - id: design
    title: "Design Overview"
    audience: theory
    sources:
      - path: docs/design.md
        sections:
          - "Overview"
          - "Architecture"
  - id: readme-page
    title: "README"
    audience: user
    sources:
      - path: README.md
        sections: all
promotions: []
"""
    (tmp_path / "wiki.yaml").write_text(config_text)
    return tmp_path


@pytest.fixture
def wiki_config(wiki_tree: Path) -> WikiConfig:
    return load_config(wiki_tree / "wiki.yaml")


# ---------- Config parsing ----------

class TestConfigParsing:
    def test_load_pages(self, wiki_config: WikiConfig):
        assert len(wiki_config.pages) == 3
        assert wiki_config.materializer_version == "0.1.0"

    def test_page_ids(self, wiki_config: WikiConfig):
        ids = [p.id for p in wiki_config.pages]
        assert ids == ["home", "design", "readme-page"]

    def test_audience_default(self, wiki_config: WikiConfig):
        readme = next(p for p in wiki_config.pages if p.id == "readme-page")
        assert readme.audience == "user"

    def test_auto_index(self, wiki_config: WikiConfig):
        home = next(p for p in wiki_config.pages if p.id == "home")
        assert home.auto_index is True

    def test_spec_hash_stable(self, wiki_config: WikiConfig):
        page = wiki_config.pages[1]
        h1 = page.spec_hash()
        h2 = page.spec_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_spec_hash_changes_on_title_change(self, wiki_config: WikiConfig):
        page = wiki_config.pages[1]
        h1 = page.spec_hash()
        page.title = "Changed Title"
        h2 = page.spec_hash()
        assert h1 != h2


# ---------- Extractor ----------

class TestExtractor:
    def test_extract_all(self, wiki_tree: Path):
        source = SourceSpec(path="README.md", sections="all")
        result = extract_sections(source, wiki_tree)
        assert "A short readme" in result
        assert "Run pip install" in result

    def test_extract_specific_sections(self, wiki_tree: Path):
        source = SourceSpec(path="docs/design.md", sections=["Overview", "Architecture"])
        result = extract_sections(source, wiki_tree)
        assert "This is the overview" in result
        assert "Arch details" in result
        assert "API docs" not in result

    def test_extract_missing_file(self, wiki_tree: Path):
        source = SourceSpec(path="nonexistent.md", sections="all")
        result = extract_sections(source, wiki_tree)
        assert "source not found" in result


# ---------- Determinism ----------

class TestDeterminism:
    def test_materialize_twice_identical(self, wiki_config: WikiConfig, wiki_tree: Path):
        """Materialize twice, assert byte-identical output."""
        out1 = wiki_tree / ".wiki1"
        out2 = wiki_tree / ".wiki2"

        m1 = materialize(wiki_config, wiki_tree, out1)
        m2 = materialize(wiki_config, wiki_tree, out2)

        assert m1.aggregate_hash == m2.aggregate_hash

        for page in wiki_config.pages:
            f1 = (out1 / f"{page.id}.md").read_bytes()
            f2 = (out2 / f"{page.id}.md").read_bytes()
            assert f1 == f2, f"Page {page.id} differs between runs"

        # Manifests should be identical
        manifest1 = (out1 / "manifest.json").read_text()
        manifest2 = (out2 / "manifest.json").read_text()
        assert manifest1 == manifest2


# ---------- Drift detection ----------

class TestDrift:
    def test_clean_after_materialize(self, wiki_config: WikiConfig, wiki_tree: Path):
        """No drift immediately after materializing."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)
        report = check_drift(wiki_config, wiki_tree, out)
        assert report.is_clean
        assert report.stale_count == 0
        assert report.orphaned_count == 0

    def test_source_change_detected(self, wiki_config: WikiConfig, wiki_tree: Path):
        """Modifying a source file triggers stale drift."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)

        # Modify a source
        design = wiki_tree / "docs" / "design.md"
        design.write_text(design.read_text() + "\nNew content added.\n")

        report = check_drift(wiki_config, wiki_tree, out)
        assert not report.is_clean
        stale_ids = [p.page_id for p in report.pages if p.status == "stale"]
        assert "design" in stale_ids

    def test_file_deletion_detected(self, wiki_config: WikiConfig, wiki_tree: Path):
        """Deleting a materialized page triggers stale drift."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)

        (out / "design.md").unlink()

        report = check_drift(wiki_config, wiki_tree, out)
        assert not report.is_clean
        design_drift = next(p for p in report.pages if p.page_id == "design")
        assert design_drift.status == "stale"
        assert any("missing" in r for r in design_drift.reasons)

    def test_file_modification_detected(self, wiki_config: WikiConfig, wiki_tree: Path):
        """Editing a materialized page on disk triggers stale drift."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)

        page_file = out / "design.md"
        page_file.write_text(page_file.read_text() + "\n<!-- hand edit -->\n")

        report = check_drift(wiki_config, wiki_tree, out)
        assert not report.is_clean

    def test_no_manifest_all_stale(self, wiki_config: WikiConfig, wiki_tree: Path):
        """When no manifest exists, all pages report stale."""
        out = wiki_tree / ".wiki"
        out.mkdir()
        report = check_drift(wiki_config, wiki_tree, out)
        assert report.stale_count == len(wiki_config.pages)

    def test_orphaned_page_detected(self, wiki_config: WikiConfig, wiki_tree: Path):
        """A page in manifest but removed from config is orphaned."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)

        # Remove last page from config
        removed = wiki_config.pages.pop()

        report = check_drift(wiki_config, wiki_tree, out)
        assert report.orphaned_count == 1
        orphaned = [p for p in report.pages if p.status == "orphaned"]
        assert orphaned[0].page_id == removed.id

    def test_version_mismatch_all_stale(self, wiki_config: WikiConfig, wiki_tree: Path):
        """Version mismatch between config and manifest → all stale."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)

        wiki_config.materializer_version = "999.0.0"
        report = check_drift(wiki_config, wiki_tree, out)
        assert report.stale_count == len(wiki_config.pages)
        assert report.version_mismatch

    def test_config_change_detected(self, wiki_config: WikiConfig, wiki_tree: Path):
        """Changing a page's config (title) triggers spec_hash drift."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)

        design = next(p for p in wiki_config.pages if p.id == "design")
        design.title = "Changed Title"

        report = check_drift(wiki_config, wiki_tree, out)
        design_drift = next(p for p in report.pages if p.page_id == "design")
        assert design_drift.status == "stale"
        assert any("config changed" in r for r in design_drift.reasons)

    def test_human_format(self, wiki_config: WikiConfig, wiki_tree: Path):
        """DriftReport.format_human() returns readable output."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)
        report = check_drift(wiki_config, wiki_tree, out)
        text = report.format_human()
        assert "drift report" in text.lower()


# ---------- Provenance (frontmatter) ----------

class TestProvenance:
    def test_frontmatter_fields_present(self, wiki_config: WikiConfig, wiki_tree: Path):
        """All required frontmatter fields are present in materialized pages."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)

        required_fields = [
            "generated:", "generated_from:", "source_hash:",
            "spec_hash:", "file_hash:", "materializer_version:",
            "theory_scope:", "audience:", "page_id:",
        ]

        for page in wiki_config.pages:
            content = (out / f"{page.id}.md").read_text()
            assert content.startswith("---"), f"{page.id} missing frontmatter start"
            for field_name in required_fields:
                assert field_name in content, f"{page.id} missing {field_name}"

    def test_no_timestamp_in_output(self, wiki_config: WikiConfig, wiki_tree: Path):
        """Determinism contract: no timestamps anywhere in output."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)

        for page in wiki_config.pages:
            content = (out / f"{page.id}.md").read_text()
            assert "generated_at" not in content

        manifest_text = (out / "manifest.json").read_text()
        assert "generated_at" not in manifest_text


# ---------- Round-trip (manifest integrity) ----------

class TestRoundTrip:
    def test_manifest_hashes_match_disk(self, wiki_config: WikiConfig, wiki_tree: Path):
        """Materialize → read manifest → verify hashes match files on disk."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)

        manifest = load_manifest(out / "manifest.json")
        assert manifest is not None

        for entry in manifest.pages:
            page_path = out / f"{entry.id}.md"
            assert page_path.exists(), f"Missing page file: {entry.id}"
            disk_hash = compute_file_hash(page_path)
            assert disk_hash == entry.file_hash, (
                f"file_hash mismatch for {entry.id}: "
                f"manifest={entry.file_hash}, disk={disk_hash}"
            )

    def test_source_hash_matches_current(
        self, wiki_config: WikiConfig, wiki_tree: Path
    ):
        """Source hashes in manifest match current source file contents."""
        out = wiki_tree / ".wiki"
        materialize(wiki_config, wiki_tree, out)

        manifest = load_manifest(out / "manifest.json")
        assert manifest is not None

        config_by_id = {p.id: p for p in wiki_config.pages}
        for entry in manifest.pages:
            page_cfg = config_by_id[entry.id]
            source_paths = [s.path for s in page_cfg.sources]
            current_hash = compute_source_hash(source_paths, wiki_tree)
            assert current_hash == entry.source_hash, (
                f"source_hash mismatch for {entry.id}"
            )

    def test_manifest_save_load_roundtrip(self, tmp_path: Path):
        """Save → load manifest preserves all fields."""
        entries = [
            PageEntry(
                id="test-page", title="Test", path=".wiki/test-page.md",
                source_hash="abcd1234abcd1234", spec_hash="efef5678efef5678",
                file_hash="1111222233334444", sources=["docs/a.md"],
                audience="user", theory_scope=False,
            ),
        ]
        original = Manifest(
            materializer_version="0.1.0",
            aggregate_hash="aaaa0000bbbb1111",
            pages=entries,
        )

        manifest_path = tmp_path / "manifest.json"
        save_manifest(original, manifest_path)
        loaded = load_manifest(manifest_path)

        assert loaded is not None
        assert loaded.materializer_version == original.materializer_version
        assert loaded.aggregate_hash == original.aggregate_hash
        assert len(loaded.pages) == 1
        assert loaded.pages[0].id == "test-page"
        assert loaded.pages[0].source_hash == "abcd1234abcd1234"
        assert loaded.pages[0].theory_scope is False

    def test_load_nonexistent_manifest(self, tmp_path: Path):
        """Loading a nonexistent manifest returns None."""
        result = load_manifest(tmp_path / "nope.json")
        assert result is None
