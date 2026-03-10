#!/usr/bin/env python3
"""Publish materialized wiki pages as a static GitHub Pages site.

Generates page-id/index.html per page for clean URLs.
Uses the same transform core as publish_wiki.py.

Usage:
    python scripts/publish_pages.py [--out-dir _site] [--dry-run] [--check-links]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import markdown
from wiki_transforms import (
    apply_common_transforms,
    load_metrics,
    load_wiki_config,
)


def page_id_to_wiki_name(page_id: str) -> str:
    """Convert page ID to GitHub Wiki filename (Title-Case)."""
    return "-".join(word.capitalize() for word in page_id.split("-"))


def rewrite_wiki_case_links(text: str, all_pages: list[dict]) -> str:
    """Rewrite Wiki-Case links (e.g. Signed-Hierarchies) to pages-relative format.

    Source markdown contains links in Wiki-Case (for the Wiki target).
    This pass converts them to pages-relative format, including fragment links
    like Glossary#anchor -> ../glossary/#anchor.
    """
    for p in all_pages:
        wiki_name = page_id_to_wiki_name(p["id"])
        pid = p["id"]
        target = f"../{pid}/" if pid != "home" else "../"
        # With fragment: Glossary#anchor -> ../glossary/#anchor
        text = re.sub(
            rf"\]\({re.escape(wiki_name)}#([^)]+)\)",
            rf"]({target}#\1)",
            text,
        )
        # Without fragment: Signed-Hierarchies -> ../signed-hierarchies/
        text = text.replace(f"]({wiki_name})", f"]({target})")

    return text


TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "wiki" / "_template.html"
)
STYLE_PATH = Path(__file__).resolve().parent.parent / "docs" / "wiki" / "style.css"
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "docs" / "wiki" / "script.js"


def pages_link_fn(page_id: str, title: str) -> str:
    """Format a markdown link for Pages target (relative paths)."""
    if page_id == "home":
        return f"[{title}](../)"
    return f"[{title}](../{page_id}/)"


def build_page_map(all_pages: list[dict]) -> dict[str, str]:
    """Build page ID -> relative link target map for Pages."""
    page_map = {}
    for p in all_pages:
        pid = p["id"]
        if pid == "home":
            page_map[pid] = "../"
        else:
            page_map[pid] = f"../{pid}/"
    return page_map


def build_sidebar_html(all_pages: list[dict], rails: dict, current_page_id: str) -> str:
    """Build sidebar HTML from wiki.yaml structure."""
    lines = []

    is_root = current_page_id == "home"
    prefix = "./" if is_root else "../"

    # Home link
    home_class = ' class="active"' if is_root else ""
    lines.append(f'<a href="{prefix}"{home_class}>Home</a>')
    lines.append("<hr>")

    # Group pages by rail
    rail_order = list(rails.keys())
    for rail_id in rail_order:
        rail_info = rails[rail_id]
        rail_name = rail_info.get("name", rail_id)
        rail_pages = [p for p in all_pages if p.get("rail") == rail_id]
        rail_pages.sort(key=lambda p: p.get("chapter", 0))
        if not rail_pages:
            continue

        lines.append(f"<h3>{rail_name}</h3>")
        lines.append("<ul>")
        for p in rail_pages:
            active = ' class="active"' if p["id"] == current_page_id else ""
            href = f"{prefix}{p['id']}/"
            lines.append(f'  <li><a href="{href}"{active}>{p["title"]}</a></li>')
        lines.append("</ul>")

    return "\n".join(lines)


def build_prev_next(
    page_config: dict, all_pages: list[dict], current_page_id: str
) -> str:
    """Build prev/next navigation HTML."""
    rail_id = page_config.get("rail")
    if not rail_id:
        return ""

    rail_pages = [p for p in all_pages if p.get("rail") == rail_id]
    rail_pages.sort(key=lambda p: p.get("chapter", 0))

    current_idx = None
    for i, p in enumerate(rail_pages):
        if p["id"] == current_page_id:
            current_idx = i
            break

    if current_idx is None:
        return ""

    parts = []
    parts.append('<nav class="prev-next">')
    if current_idx > 0:
        prev_p = rail_pages[current_idx - 1]
        parts.append(
            f'  <a href="../{prev_p["id"]}/" class="prev">&larr; {prev_p["title"]}</a>'
        )
    else:
        parts.append('  <span class="prev"></span>')

    if current_idx < len(rail_pages) - 1:
        next_p = rail_pages[current_idx + 1]
        parts.append(
            f'  <a href="../{next_p["id"]}/" class="next">{next_p["title"]} &rarr;</a>'
        )
    else:
        parts.append('  <span class="next"></span>')

    parts.append("</nav>")
    return "\n".join(parts)


def md_to_html(md_text: str) -> str:
    """Convert markdown to HTML with fenced code blocks and tables."""
    return markdown.markdown(
        md_text,
        extensions=["fenced_code", "tables", "toc"],
        output_format="html",
    )


def render_page(
    template: str,
    title: str,
    content_html: str,
    sidebar_html: str,
    prev_next_html: str,
    description: str,
    is_root: bool = False,
) -> str:
    """Render a full HTML page from template."""
    html = template
    html = html.replace("{{title}}", title)
    html = html.replace("{{description}}", description)
    html = html.replace("{{sidebar}}", sidebar_html)
    html = html.replace("{{content}}", content_html)
    html = html.replace("{{prev_next}}", prev_next_html)
    # Root page (index.html) uses ./ prefix; subpages use ../
    if is_root:
        html = html.replace('href="../"', 'href="./"')
        html = html.replace('href="../style.css"', 'href="./style.css"')
        html = html.replace('src="../script.js"', 'src="./script.js"')
        html = html.replace('href="../glossary/"', 'href="./glossary/"')
        html = html.replace('href="../getting-started/"', 'href="./getting-started/"')
    return html


def extract_description(md_text: str) -> str:
    """Extract the first non-heading, non-empty paragraph as meta description."""
    for line in md_text.split("\n"):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(">") or s.startswith("```"):
            continue
        if s.startswith("---"):
            continue
        # Strip markdown formatting
        desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
        desc = re.sub(r"[*_`]", "", desc)
        if len(desc) > 20:
            return desc[:160]
    return "ModelAtlas documentation"


def generate_sitemap(all_pages: list[dict], base_url: str) -> str:
    """Generate sitemap.xml content."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
    # Home
    lines.append(f"  <url><loc>{base_url}</loc></url>")
    for p in all_pages:
        if p["id"] == "home":
            continue
        lines.append(f"  <url><loc>{base_url}{p['id']}/</loc></url>")
    lines.append("</urlset>")
    return "\n".join(lines)


def check_links(out_dir: Path, all_pages: list[dict]) -> list[str]:
    """Check for broken internal links in generated HTML files."""
    errors = []
    for html_file in out_dir.rglob("*.html"):
        content = html_file.read_text()
        for match in re.finditer(r'href="([^"]*)"', content):
            href = match.group(1)
            # Skip external links, anchors-only, assets
            if href.startswith(("http://", "https://", "#", "mailto:")):
                continue
            if href.endswith((".css", ".js")):
                continue

            # Strip fragment before resolving
            href_path = href.split("#")[0]
            if not href_path or href_path == "./":
                continue

            # Resolve relative to the HTML file's directory
            target = (html_file.parent / href_path).resolve()

            # Valid if it's a directory with index.html, or is the out_dir root
            if target == out_dir.resolve():
                continue
            if target.is_dir() and (target / "index.html").exists():
                continue
            if target.is_file():
                continue

            rel = html_file.relative_to(out_dir)
            errors.append(f"  {rel}: broken link -> {href}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish wiki as GitHub Pages static site"
    )
    parser.add_argument(
        "--out-dir",
        default="_site",
        help="Output directory (default: _site)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (default: auto-detect)",
    )
    parser.add_argument(
        "--base-url",
        default="https://rohanv.me/ModelAtlas/",
        help="Base URL for sitemap (default: https://rohanv.me/ModelAtlas/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="Check for broken links after generation",
    )
    args = parser.parse_args()

    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        repo_root = Path(__file__).resolve().parent.parent

    out_dir = Path(args.out_dir)
    materialized_dir = repo_root / ".wiki"

    if not materialized_dir.exists():
        print(
            "ERROR: .wiki/ not found. Run `python -m model_atlas.wiki materialize` first."
        )
        return 1

    # Load config and metrics
    config = load_wiki_config(repo_root)
    metrics = load_metrics(repo_root)
    rails = config.get("rails", {})
    all_pages = config.get("pages", [])

    # Load template
    if not TEMPLATE_PATH.exists():
        print(f"ERROR: Template not found at {TEMPLATE_PATH}")
        return 1
    template = TEMPLATE_PATH.read_text()

    # Build page map for relative links
    page_map = build_page_map(all_pages)

    if not args.dry_run:
        # Clean and create output dir
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

    # Process each page
    for page_config in all_pages:
        page_id = page_config["id"]
        source_path = materialized_dir / f"{page_id}.md"

        if not source_path.exists():
            print(f"  SKIP {page_id} (not materialized)")
            continue

        # Apply shared transforms (breadcrumb links use pages format)
        md_content = apply_common_transforms(
            source_path,
            page_config,
            metrics,
            rails,
            all_pages,
            page_map,
            link_fn=pages_link_fn,
        )

        # Second pass: rewrite any remaining Wiki-Case links from source content
        md_content = rewrite_wiki_case_links(md_content, all_pages)

        # Root page: convert ../ prefix to ./ since index.html is at site root
        if page_id == "home":
            md_content = md_content.replace("](../", "](./")

        # Convert markdown to HTML
        content_html = md_to_html(md_content)
        sidebar_html = build_sidebar_html(all_pages, rails, page_id)
        prev_next_html = build_prev_next(page_config, all_pages, page_id)
        description = extract_description(md_content)

        page_html = render_page(
            template,
            title=page_config["title"],
            content_html=content_html,
            sidebar_html=sidebar_html,
            prev_next_html=prev_next_html,
            description=description,
            is_root=(page_id == "home"),
        )

        # Write as page-id/index.html (home -> index.html at root)
        if page_id == "home":
            dest = out_dir / "index.html"
        else:
            dest = out_dir / page_id / "index.html"

        if args.dry_run:
            print(f"  {page_id} -> {dest.relative_to(out_dir)}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(page_html)
            print(f"  {page_id} -> {dest.relative_to(out_dir)}")

    if not args.dry_run:
        # Copy static assets
        if STYLE_PATH.exists():
            shutil.copy2(STYLE_PATH, out_dir / "style.css")
            print("  style.css")
        if SCRIPT_PATH.exists():
            shutil.copy2(SCRIPT_PATH, out_dir / "script.js")
            print("  script.js")

        # Generate sitemap
        sitemap = generate_sitemap(all_pages, args.base_url)
        (out_dir / "sitemap.xml").write_text(sitemap)
        print("  sitemap.xml")

        # robots.txt
        robots = f"User-agent: *\nAllow: /\nSitemap: {args.base_url}sitemap.xml\n"
        (out_dir / "robots.txt").write_text(robots)
        print("  robots.txt")

        # .nojekyll
        (out_dir / ".nojekyll").write_text("")
        print("  .nojekyll")

    # Link check
    if args.check_links and not args.dry_run:
        print("\nChecking links...")
        errors = check_links(out_dir, all_pages)
        if errors:
            print(f"Found {len(errors)} broken link(s):")
            for err in errors:
                print(err)
            return 1
        else:
            print("All links OK.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
