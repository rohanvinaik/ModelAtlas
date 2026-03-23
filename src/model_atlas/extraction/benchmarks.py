"""Benchmark table extraction and anchor derivation from model card text.

Parses markdown pipe-delimited tables that contain benchmark results,
returning metadata entries for each recognized benchmark score.
Also derives QUALITY anchors when benchmark scores exceed thresholds.
"""

from __future__ import annotations

import re

from .deterministic import AnchorTag

# Keywords that indicate a benchmark table header
_BENCHMARK_KEYWORDS = re.compile(
    r"benchmark|eval|score|accuracy|perplexity|mmlu|hellaswag|winogrande|arc|gsm8k|truthfulqa|humaneval",
    re.IGNORECASE,
)

# Matches a markdown table row: | cell1 | cell2 | ... |
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")

# Matches the separator row: |---|---|
# Single character class avoids overlapping quantifiers (ReDoS-safe).
# Lookahead requires at least one dash so we don't match empty pipes.
_SEPARATOR_ROW = re.compile(r"^\s*\|(?=[-:\s|]*-)[-:\s|]+\|\s*$")


def _extract_row_score(cells: list[str]) -> tuple[str, str] | None:
    """Extract benchmark name and score from a parsed table row.

    Returns (name, score_text) or None if the row has no valid score.
    """
    if len(cells) < 2 or not cells[0]:
        return None
    name = cells[0].lower().replace(" ", "-")
    for cell in cells[1:]:
        if cell and re.search(r"\d", cell):
            return name, cell
    return None


def _is_benchmark_table(header_cells: list[str]) -> bool:
    """Check if a table header contains benchmark-related keywords."""
    return bool(_BENCHMARK_KEYWORDS.search(" ".join(header_cells)))


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
        header_match = _TABLE_ROW.match(lines[i])
        if not header_match:
            i += 1
            continue

        header_cells = [c.strip() for c in header_match.group(1).split("|")]

        if i + 1 >= len(lines) or not _SEPARATOR_ROW.match(lines[i + 1]):
            i += 1
            continue

        if not _is_benchmark_table(header_cells):
            i += 2
            continue

        i += 2
        while i < len(lines):
            row_match = _TABLE_ROW.match(lines[i])
            if not row_match:
                break
            cells = [c.strip() for c in row_match.group(1).split("|")]
            row_score = _extract_row_score(cells)
            if row_score:
                results[f"benchmark:{row_score[0]}"] = (row_score[1], "str")
            i += 1

    return results


# Thresholds for deriving QUALITY anchors from benchmark scores
_BENCHMARK_THRESHOLDS: dict[str, tuple[str, float]] = {
    "mmlu": ("high-mmlu", 70.0),
    "humaneval": ("strong-humaneval", 40.0),
    "gsm8k": ("strong-gsm8k", 60.0),
}


def _parse_score(raw: str) -> float | None:
    """Extract a numeric score from a raw benchmark value string."""
    match = re.search(r"(\d+(?:\.\d+)?)", raw)
    if match:
        return float(match.group(1))
    return None


def derive_benchmark_anchors(
    benchmarks: dict[str, tuple[str, str]],
) -> list[AnchorTag]:
    """Derive QUALITY anchors from benchmark scores exceeding thresholds.

    Returns AnchorTag list with confidence=0.75 for each threshold met.
    """
    anchors: list[AnchorTag] = []
    for bench_key, (raw_score, _) in benchmarks.items():
        # bench_key is like "benchmark:mmlu"
        name = bench_key.split(":", 1)[-1] if ":" in bench_key else bench_key
        if name not in _BENCHMARK_THRESHOLDS:
            continue
        anchor_label, threshold = _BENCHMARK_THRESHOLDS[name]
        score = _parse_score(raw_score)
        if score is not None and score >= threshold:
            anchors.append(AnchorTag(anchor_label, "QUALITY", 0.75))
    return anchors
