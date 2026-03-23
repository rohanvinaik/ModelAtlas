"""Section extraction from markdown files."""

from __future__ import annotations

import re
from pathlib import Path

from .config import SourceSpec


def extract_sections(source: SourceSpec, repo_root: Path) -> str:
    """Extract content from a source file according to its spec.

    Returns the extracted text as a string.
    """
    file_path = repo_root / source.path
    if not file_path.exists():
        return f"<!-- source not found: {source.path} -->\n"

    text = file_path.read_text()

    if source.extract == "docstrings":
        return _extract_docstrings(text)

    if source.sections == "all":
        return _strip_yaml_frontmatter(text)

    sections = (
        source.sections if isinstance(source.sections, list) else [source.sections]
    )
    return _extract_heading_sections(text, sections)


def _strip_yaml_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (--- delimited) from the start of a file."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3 :].lstrip("\n")
    return text


def _extract_heading_sections(text: str, section_titles: list[str]) -> str:
    """Extract sections by heading title (case-insensitive prefix match).

    Captures everything from the matched heading to the next heading
    at the same or higher level.
    """
    lines = text.split("\n")
    sections = _parse_headings(lines)
    result_parts = []

    for target in section_titles:
        target_lower = target.lower().strip()
        for heading_level, heading_title, start, end in sections:
            # Prefix match, case-insensitive, strip numbering like "1. " or "2.1 "
            clean_title = re.sub(r"^\d+(\.\d+)*\.?\s*", "", heading_title).strip()
            if clean_title.lower().startswith(
                target_lower
            ) or heading_title.lower().startswith(target_lower):
                result_parts.append("\n".join(lines[start:end]))
                break

    return "\n\n".join(result_parts)


def _parse_headings(lines: list[str]) -> list[tuple[int, str, int, int]]:
    """Parse markdown headings and their line ranges.

    Returns list of (level, title, start_line, end_line).
    """
    headings: list[tuple[int, str, int]] = []

    for i, line in enumerate(lines):
        stripped = line.lstrip("#")
        hashes = len(line) - len(stripped)
        if 1 <= hashes <= 6 and stripped and stripped[0] == " ":
            title = stripped.strip()
            if title:
                headings.append((hashes, title, i))

    result = []
    for idx, (level, title, start) in enumerate(headings):
        # Find end: next heading at same or higher level
        end = len(lines)
        for future_level, _, future_start in headings[idx + 1 :]:
            if future_level <= level:
                end = future_start
                break
        result.append((level, title, start, end))

    return result


def _extract_docstrings(text: str) -> str:
    """Extract module and function/class docstrings from Python source."""
    parts = []
    in_docstring = False
    quote_char = ""
    current: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()

        if not in_docstring:
            for q in ['"""', "'''"]:
                if q in stripped:
                    in_docstring = True
                    quote_char = q
                    after = stripped.split(q, 1)[1]
                    if q in after:
                        # Single-line docstring
                        parts.append(after.split(q, 1)[0].strip())
                        in_docstring = False
                    else:
                        current = [after] if after else []
                    break
        else:
            if quote_char in stripped:
                before = stripped.split(quote_char, 1)[0]
                if before:
                    current.append(before)
                parts.append("\n".join(current).strip())
                current = []
                in_docstring = False
            else:
                current.append(line.rstrip())

    return "\n\n".join(p for p in parts if p)
