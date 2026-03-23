"""Shared transforms for wiki/pages publishing.

Both publish_wiki.py and publish_pages.py use these functions so that
frontmatter stripping, H1 removal, metric interpolation, breadcrumbs,
and link rewriting never diverge between targets.
"""

from __future__ import annotations

import math
from collections.abc import Callable
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
    """Remove all leading H1 headings.

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
                continue
            if s.startswith("# ") and not s.startswith("## "):
                stripped_any = True
                continue
            in_leading = False
        result.append(line)
    if stripped_any:
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
    page_config: dict,
    rails: dict,
    all_pages: list[dict],
    link_fn: Callable,
) -> str:
    """Build a breadcrumb line for a page.

    link_fn(page_id, title) -> markdown link string.
    This lets each publisher format links for its target.
    """
    rail_id = page_config.get("rail")
    if not rail_id:
        return ""

    rail_info = rails.get(rail_id, {})
    rail_name = rail_info.get("name", rail_id)

    rail_pages = [p for p in all_pages if p.get("rail") == rail_id]
    rail_pages.sort(key=lambda p: p.get("chapter", 0))
    total = len(rail_pages)
    chapter = page_config.get("chapter", 1)

    prereqs = page_config.get("prerequisites", [])
    if prereqs:
        prereq_links = []
        for pid in prereqs:
            title = pid
            for p in all_pages:
                if p["id"] == pid:
                    title = p["title"]
                    break
            prereq_links.append(link_fn(pid, title))
        prereq_str = "Prerequisites: " + ", ".join(prereq_links)
    else:
        prereq_str = "Prerequisites: none"

    return (
        f"> **{rail_name}** · Chapter {chapter} of {total} · "
        f"{prereq_str} · ~{{read_time}} min read\n\n"
    )


def rewrite_links(text: str, page_map: dict[str, str]) -> str:
    """Rewrite internal links from page-id to target names."""
    for pid, target_name in page_map.items():
        text = text.replace(f"]({pid})", f"]({target_name})")
        text = text.replace(f"]({pid}.md)", f"]({target_name})")
    return text


def apply_common_transforms(
    source_path: Path,
    page_config: dict,
    metrics: dict[str, str],
    rails: dict,
    all_pages: list[dict],
    page_map: dict[str, str],
    link_fn: Callable,
) -> str:
    """Apply the shared transform pipeline to a page.

    Returns the fully transformed markdown text.
    """
    text = source_path.read_text()
    text = strip_frontmatter(text)
    text = strip_leading_h1(text)
    text = interpolate_metrics(text, metrics)

    breadcrumb = build_breadcrumb(page_config, rails, all_pages, link_fn)
    if breadcrumb:
        read_time = compute_read_time(text)
        breadcrumb = breadcrumb.replace("{read_time}", str(read_time))
        text = breadcrumb + text

    text = rewrite_links(text, page_map)
    return text
