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
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml


def load_metrics(repo_root: Path) -> dict[str, str]:
    """Load volatile metrics from _metrics.yaml."""
    metrics_path = repo_root / "docs" / "wiki" / "_metrics.yaml"
    if not metrics_path.exists():
        return {}
    data = yaml.safe_load(metrics_path.read_text())
    return {k: str(v) for k, v in data.items()}


def load_wiki_config(repo_root: Path) -> dict:
    """Load wiki.yaml with rail/chapter/prerequisites metadata."""
    return yaml.safe_load((repo_root / "wiki.yaml").read_text())


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (--- delimited) from start of file."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3 :].lstrip("\n")
    return text


def strip_leading_h1(text: str) -> str:
    """Remove all leading H1 headings (GitHub Wiki shows filename as title).

    The materializer may produce duplicate H1s: one from _render_page_body
    and one from the source file itself. Strip all of them.
    """
    lines = text.split("\n")
    result = []
    stripped_any = False
    in_leading = True
    for line in lines:
        s = line.strip()
        if in_leading:
            if not s:
                continue  # skip leading blanks
            if s.startswith("# ") and not s.startswith("## "):
                stripped_any = True
                continue  # skip H1
            in_leading = False
        result.append(line)
    if stripped_any:
        # Remove any leading blank lines from result
        while result and result[0].strip() == "":
            result = result[1:]
    return "\n".join(result) if stripped_any else text


def interpolate_metrics(text: str, metrics: dict[str, str]) -> str:
    """Replace {{key}} placeholders with metric values."""
    for key, value in metrics.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def compute_read_time(text: str) -> int:
    """Estimate read time in minutes (200 wpm for technical content)."""
    words = len(text.split())
    return max(1, math.ceil(words / 200))


def build_breadcrumb(
    page_config: dict, rails: dict, all_pages: list[dict]
) -> str:
    """Build a 'You are here' breadcrumb line for a page."""
    rail_id = page_config.get("rail")
    if not rail_id:
        return ""

    rail_info = rails.get(rail_id, {})
    rail_name = rail_info.get("name", rail_id)

    # Count pages in this rail
    rail_pages = [p for p in all_pages if p.get("rail") == rail_id]
    rail_pages.sort(key=lambda p: p.get("chapter", 0))
    total = len(rail_pages)
    chapter = page_config.get("chapter", 1)

    # Prerequisites
    prereqs = page_config.get("prerequisites", [])
    if prereqs:
        prereq_links = []
        for pid in prereqs:
            wiki_name = page_id_to_wiki_name(pid)
            # Find the title
            title = pid
            for p in all_pages:
                if p["id"] == pid:
                    title = p["title"]
                    break
            prereq_links.append(f"[{title}]({wiki_name})")
        prereq_str = "Prerequisites: " + ", ".join(prereq_links)
    else:
        prereq_str = "Prerequisites: none"

    return (
        f"> **{rail_name}** · Chapter {chapter} of {total} · "
        f"{prereq_str} · ~{{read_time}} min read\n\n"
    )


def page_id_to_wiki_name(page_id: str) -> str:
    """Convert page ID to GitHub Wiki filename (Title-Case)."""
    return "-".join(word.capitalize() for word in page_id.split("-"))


PAGE_MAP = {}  # populated at runtime


def rewrite_links(text: str, page_map: dict[str, str]) -> str:
    """Rewrite internal links from page-id to Wiki-Name format."""
    for pid, wname in page_map.items():
        text = text.replace(f"]({pid})", f"]({wname})")
        text = text.replace(f"]({pid}.md)", f"]({wname})")
    return text


def process_page(
    source_path: Path,
    page_config: dict,
    metrics: dict[str, str],
    rails: dict,
    all_pages: list[dict],
    page_map: dict[str, str],
) -> str:
    """Apply all transforms to a single page."""
    text = source_path.read_text()

    # 1. Strip frontmatter
    text = strip_frontmatter(text)

    # 2. Strip duplicate H1
    text = strip_leading_h1(text)

    # 3. Interpolate metrics
    text = interpolate_metrics(text, metrics)

    # 4. Build and inject breadcrumb
    breadcrumb = build_breadcrumb(page_config, rails, all_pages)
    if breadcrumb:
        read_time = compute_read_time(text)
        breadcrumb = breadcrumb.replace("{read_time}", str(read_time))
        text = breadcrumb + text

    # 5. Rewrite internal links
    text = rewrite_links(text, page_map)

    return text


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
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--push", action="store_true", help="Git commit and push after writing")
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
        print("ERROR: .wiki/ not found. Run `python -m model_atlas.wiki materialize` first.")
        return 1

    if not wiki_dir.exists():
        print(f"ERROR: Wiki repo not found at {wiki_dir}")
        print("Clone it: git clone https://github.com/rohanvinaik/ModelAtlas.wiki.git /tmp/ModelAtlas.wiki")
        return 1

    # Load config and metrics
    config = load_wiki_config(repo_root)
    metrics = load_metrics(repo_root)
    rails = config.get("rails", {})
    all_pages = config.get("pages", [])

    # Build page ID -> wiki name map
    page_map = {p["id"]: page_id_to_wiki_name(p["id"]) for p in all_pages}

    # Remove old pages that aren't in the new config
    existing_files = set(f.name for f in wiki_dir.glob("*.md") if not f.name.startswith("_"))
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

        content = process_page(
            source_path, page_config, metrics, rails, all_pages, page_map
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
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=wiki_dir
        )
        if result.returncode == 0:
            print("\nNo changes to push.")
        else:
            subprocess.run(
                ["git", "commit", "-m", msg], cwd=wiki_dir, check=True
            )
            subprocess.run(["git", "push"], cwd=wiki_dir, check=True)
            print("\nPushed to wiki repo.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
