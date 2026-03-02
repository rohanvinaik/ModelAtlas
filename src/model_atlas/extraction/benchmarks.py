"""Benchmark table extraction from model card text.

Parses markdown pipe-delimited tables that contain benchmark results,
returning metadata entries for each recognized benchmark score.
"""

from __future__ import annotations

import re

# Keywords that indicate a benchmark table header
_BENCHMARK_KEYWORDS = re.compile(
    r"benchmark|eval|score|accuracy|perplexity|mmlu|hellaswag|winogrande|arc|gsm8k|truthfulqa|humaneval",
    re.IGNORECASE,
)

# Matches a markdown table row: | cell1 | cell2 | ... |
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")

# Matches the separator row: |---|---|
_SEPARATOR_ROW = re.compile(r"^\s*\|[\s:]*-+[\s:|-]*\|\s*$")


def extract_benchmarks(card_text: str) -> dict[str, tuple[str, str]]:
    """Extract benchmark scores from markdown tables in model card text.

    Returns a dict of metadata entries: {"benchmark:<name>": (score, "str")}.
    Only tables whose header row contains a benchmark keyword are parsed.
    """
    if not card_text:
        return {}

    lines = card_text.split("\n")
    results: dict[str, tuple[str, str]] = {}
    i = 0

    while i < len(lines):
        # Look for a table header row
        header_match = _TABLE_ROW.match(lines[i])
        if not header_match:
            i += 1
            continue

        header_cells = [c.strip() for c in header_match.group(1).split("|")]

        # Check if next line is a separator
        if i + 1 >= len(lines) or not _SEPARATOR_ROW.match(lines[i + 1]):
            i += 1
            continue

        # Check if header contains benchmark keywords
        header_text = " ".join(header_cells)
        if not _BENCHMARK_KEYWORDS.search(header_text):
            i += 2  # skip header + separator
            continue

        # Parse data rows
        i += 2  # skip header + separator
        while i < len(lines):
            row_match = _TABLE_ROW.match(lines[i])
            if not row_match:
                break
            cells = [c.strip() for c in row_match.group(1).split("|")]
            if len(cells) >= 2 and cells[0]:
                # First cell = benchmark name, remaining cells = scores
                name = cells[0].lower().replace(" ", "-")
                # Take the first non-empty score cell
                for cell in cells[1:]:
                    if cell and re.search(r"\d", cell):
                        key = f"benchmark:{name}"
                        results[key] = (cell, "str")
                        break
            i += 1

    return results
