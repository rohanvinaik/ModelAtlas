#!/usr/bin/env python3
"""Publish materialized wiki pages to the GitHub Wiki repo.

Programmatic transforms applied during publish:
1. Strip YAML frontmatter (GitHub Wiki renders it as raw text)
2. Strip duplicate H1 (GitHub Wiki shows filename as page title)
3. Interpolate volatile metrics from _metrics.yaml
4. Inject breadcrumb header (rail, chapter, prerequisites, read time)
5. Rewrite internal links to GitHub Wiki naming convention
6. Copy _Sidebar.md and _Footer.md

Usage:
    python scripts/publish_wiki.py [--wiki-dir /path/to/wiki/repo] [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from wiki_transforms import (
    apply_common_transforms,
    interpolate_metrics,
    load_metrics,
    load_wiki_config,
    rewrite_links,
)


def page_id_to_wiki_name(page_id: str) -> str:
    """Convert page ID to GitHub Wiki filename (Title-Case)."""
    return "-".join(word.capitalize() for word in page_id.split("-"))


def wiki_link_fn(page_id: str, title: str) -> str:
    """Format a markdown link for GitHub Wiki target."""
    return f"[{title}]({page_id_to_wiki_name(page_id)})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish wiki to GitHub Wiki repo")
    parser.add_argument(
        "--wiki-dir",
        default="/tmp/ModelAtlas.wiki",
        help="Path to cloned wiki repo (default: /tmp/ModelAtlas.wiki)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument(
        "--push", action="store_true", help="Git commit and push after writing"
    )
    parser.add_argument("--message", default=None, help="Commit message (with --push)")
    args = parser.parse_args()

    # Find repo root
    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        repo_root = Path(__file__).resolve().parent.parent

    wiki_dir = Path(args.wiki_dir)
    materialized_dir = repo_root / ".wiki"
    docs_wiki = repo_root / "docs" / "wiki"

    if not materialized_dir.exists():
        print(
            "ERROR: .wiki/ not found. Run `python -m model_atlas.wiki materialize` first."
        )
        return 1

    if not wiki_dir.exists():
        print(f"ERROR: Wiki repo not found at {wiki_dir}")
        print(
            "Clone it: git clone https://github.com/rohanvinaik/ModelAtlas.wiki.git /tmp/ModelAtlas.wiki"
        )
        return 1

    # Load config and metrics
    config = load_wiki_config(repo_root)
    metrics = load_metrics(repo_root)
    rails = config.get("rails", {})
    all_pages = config.get("pages", [])

    # Build page ID -> wiki name map
    page_map = {p["id"]: page_id_to_wiki_name(p["id"]) for p in all_pages}

    # Remove old pages that aren't in the new config
    existing_files = set(
        f.name for f in wiki_dir.glob("*.md") if not f.name.startswith("_")
    )
    new_files = set(f"{page_map[p['id']]}.md" for p in all_pages)
    for old_file in existing_files - new_files:
        if args.dry_run:
            print(f"  WOULD REMOVE {old_file}")
        else:
            (wiki_dir / old_file).unlink()
            print(f"  REMOVED {old_file}")

    # Process each page
    for page_config in all_pages:
        page_id = page_config["id"]
        wiki_name = page_map[page_id]
        source_path = materialized_dir / f"{page_id}.md"

        if not source_path.exists():
            print(f"  SKIP {page_id} (not materialized)")
            continue

        content = apply_common_transforms(
            source_path,
            page_config,
            metrics,
            rails,
            all_pages,
            page_map,
            link_fn=wiki_link_fn,
        )

        dest = wiki_dir / f"{wiki_name}.md"
        if args.dry_run:
            lines = content.count("\n")
            print(f"  {page_id} -> {wiki_name}.md ({lines} lines)")
        else:
            dest.write_text(content)
            print(f"  {page_id} -> {wiki_name}.md")

    # Copy special files (_Sidebar.md, _Footer.md)
    for special in ["_Sidebar.md", "_Footer.md"]:
        src = docs_wiki / special
        if src.exists():
            content = src.read_text()
            content = rewrite_links(content, page_map)
            content = interpolate_metrics(content, metrics)
            dest = wiki_dir / special
            if args.dry_run:
                print(f"  {special} (special)")
            else:
                dest.write_text(content)
                print(f"  {special}")

    # Push if requested
    if args.push and not args.dry_run:
        msg = args.message or "Update wiki from materializer"
        subprocess.run(["git", "add", "-A"], cwd=wiki_dir, check=True)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=wiki_dir)
        if result.returncode == 0:
            print("\nNo changes to push.")
        else:
            subprocess.run(["git", "commit", "-m", msg], cwd=wiki_dir, check=True)
            subprocess.run(["git", "push"], cwd=wiki_dir, check=True)
            print("\nPushed to wiki repo.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
