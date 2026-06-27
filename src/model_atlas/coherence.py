"""Coherence audit — periodic health report for the canonical network.

Read-only checks that surface drift over time:

- **Bank orthogonality** (§22, §39) — high correlations between bank
  positions suggest the dimensions are no longer independent.
- **NULL coverage per bank** (§18) — banks with growing NULL fractions
  point at extraction gaps.
- **Anchor orphans** (§44) — anchors with zero assignments are dead
  weight; either remove or expand to make useful.
- **Anchor oversaturation** (§44) — anchors assigned to >50% of models
  no longer discriminate. Candidates for splitting into sub-anchors.
- **Uncited canonical writes** (§43) — audit-log entries whose
  ``reason`` is too thin to trust (e.g. ``"update"``, ``"fix"``).

Every check is *read-only*. The audit is a maintenance loop, not an
automated rollback — output is a structured dict for the caller to act on.

CLI usage (via ``model_atlas.coherence``)::

    python -m model_atlas.coherence            # full report to stdout
    python -m model_atlas.coherence --json     # machine-readable

For programmatic use::

    from model_atlas import coherence
    report = coherence.run_audit(conn)
    if report["anchors"]["oversaturated"]:
        ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .admin import DEFAULT_AUDIT_LOG_PATH, read_audit_log
from .db import BANKS, get_connection

logger = logging.getLogger(__name__)


# Reasons containing these tokens (case-insensitive) are too thin to count
# as a sourced citation. From doc §7 examples of bad reasons.
_THIN_REASON_TOKENS: frozenset[str] = frozenset(
    {"update", "fix", "correct", "improve", "tweak", "edit", "change"}
)


@dataclass
class CoherenceReport:
    """Top-level report structure returned by run_audit."""

    bank_correlations: dict[str, float]
    bank_correlations_suspicious: dict[str, float]
    null_coverage_per_bank: dict[str, dict[str, int | float]]
    anchor_orphans: list[str]
    anchor_oversaturated: list[dict[str, Any]]
    anchor_total: int
    model_total: int
    uncited_canonical_writes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_correlations": self.bank_correlations,
            "bank_correlations_suspicious": self.bank_correlations_suspicious,
            "null_coverage_per_bank": self.null_coverage_per_bank,
            "anchor_orphans": self.anchor_orphans,
            "anchor_oversaturated": self.anchor_oversaturated,
            "anchor_total": self.anchor_total,
            "model_total": self.model_total,
            "uncited_canonical_writes": self.uncited_canonical_writes,
        }


def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Return Pearson r between two equal-length lists, or None if undefined."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    try:
        sx = statistics.stdev(xs)
        sy = statistics.stdev(ys)
    except statistics.StatisticsError:
        return None
    if sx == 0 or sy == 0:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    n = len(xs)
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (n - 1)
    return cov / (sx * sy)


def _signed_position(row: sqlite3.Row) -> float:
    """Convert (path_sign, path_depth) to a signed scalar."""
    return float(row["path_sign"]) * float(row["path_depth"])


def check_bank_correlations(
    conn: sqlite3.Connection,
    threshold: float = 0.7,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute pairwise Pearson r between bank positions.

    Returns ``(all_correlations, suspicious)`` where suspicious is the
    subset with ``|r| >= threshold``. Pair keys are ``"BANK_A__BANK_B"``
    with alphabetical ordering.
    """
    # Pull model_id -> {bank: signed} dict
    rows = conn.execute(
        "SELECT model_id, bank, path_sign, path_depth FROM model_positions"
    ).fetchall()
    by_model: dict[str, dict[str, float]] = {}
    for r in rows:
        by_model.setdefault(r["model_id"], {})[r["bank"]] = _signed_position(r)

    correlations: dict[str, float] = {}
    suspicious: dict[str, float] = {}
    bank_list = sorted(BANKS)
    for i, b1 in enumerate(bank_list):
        for b2 in bank_list[i + 1 :]:
            xs: list[float] = []
            ys: list[float] = []
            for m in by_model.values():
                if b1 in m and b2 in m:
                    xs.append(m[b1])
                    ys.append(m[b2])
            corr = _pearson_correlation(xs, ys)
            if corr is None:
                continue
            pair_key = f"{b1}__{b2}"
            correlations[pair_key] = round(corr, 4)
            if abs(corr) >= threshold:
                suspicious[pair_key] = round(corr, 4)
    return correlations, suspicious


def check_null_coverage(
    conn: sqlite3.Connection,
) -> dict[str, dict[str, int | float]]:
    """For each bank, return ``{models_with: n, missing: n, coverage_pct: %}``.

    "Missing" = models that exist but have no row in model_positions for
    this bank. This is the doc's NULL coverage metric (§18, §44).
    """
    total = int(conn.execute("SELECT COUNT(*) FROM models").fetchone()[0])
    out: dict[str, dict[str, int | float]] = {}
    for bank in BANKS:
        with_pos = int(
            conn.execute(
                "SELECT COUNT(DISTINCT model_id) FROM model_positions WHERE bank = ?",
                (bank,),
            ).fetchone()[0]
        )
        missing = total - with_pos
        coverage_pct = round(100.0 * with_pos / total, 2) if total > 0 else 0.0
        out[bank] = {
            "models_with": with_pos,
            "missing": missing,
            "coverage_pct": coverage_pct,
        }
    return out


def check_anchor_orphans(conn: sqlite3.Connection) -> list[str]:
    """Anchors with zero model assignments. Labels in alphabetical order."""
    rows = conn.execute(
        """
        SELECT a.label
        FROM anchors a
        LEFT JOIN model_anchors ma ON ma.anchor_id = a.anchor_id
        GROUP BY a.anchor_id
        HAVING COUNT(ma.model_id) = 0
        ORDER BY a.label
        """
    ).fetchall()
    return [r["label"] for r in rows]


def check_anchor_oversaturation(
    conn: sqlite3.Connection,
    threshold_pct: float = 50.0,
) -> list[dict[str, Any]]:
    """Anchors assigned to more than ``threshold_pct`` of all models.

    Doc §44 — once an anchor is assigned to a majority of entities it no
    longer discriminates and should be split into sub-anchors.
    """
    total = int(conn.execute("SELECT COUNT(*) FROM models").fetchone()[0])
    if total == 0:
        return []
    cutoff = int(total * threshold_pct / 100)
    rows = conn.execute(
        """
        SELECT a.label, a.bank, COUNT(DISTINCT ma.model_id) AS n
        FROM anchors a
        JOIN model_anchors ma ON ma.anchor_id = a.anchor_id
        GROUP BY a.anchor_id
        HAVING n > ?
        ORDER BY n DESC, a.label
        """,
        (cutoff,),
    ).fetchall()
    return [
        {
            "label": r["label"],
            "bank": r["bank"],
            "assignments": int(r["n"]),
            "share_pct": round(100.0 * int(r["n"]) / total, 2),
        }
        for r in rows
    ]


def check_uncited_canonical_writes(
    audit_log_path: Path | None = None,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Audit log entries with thin/missing reasons. Returns up to ``limit``."""
    path = audit_log_path or DEFAULT_AUDIT_LOG_PATH
    entries = read_audit_log(path)
    flagged: list[dict[str, Any]] = []
    for entry in entries:
        reason = entry.get("reason") or ""
        stripped = reason.strip().lower()
        if not stripped:
            flagged.append({"reason": reason, "entry": entry, "issue": "empty"})
            continue
        if stripped in _THIN_REASON_TOKENS:
            flagged.append(
                {"reason": reason, "entry": entry, "issue": "thin_single_word"}
            )
            continue
        # Single short token check: too short to source anything
        tokens = stripped.split()
        if len(tokens) == 1 and len(stripped) < 8:
            flagged.append(
                {"reason": reason, "entry": entry, "issue": "too_short"}
            )
        if len(flagged) >= limit:
            break
    return flagged[:limit]


def run_audit(
    conn: sqlite3.Connection,
    *,
    correlation_threshold: float = 0.7,
    oversaturation_pct: float = 50.0,
    audit_log_path: Path | None = None,
) -> CoherenceReport:
    """Run the full audit and return a structured report."""
    all_corr, suspicious = check_bank_correlations(conn, threshold=correlation_threshold)
    nulls = check_null_coverage(conn)
    orphans = check_anchor_orphans(conn)
    oversat = check_anchor_oversaturation(conn, threshold_pct=oversaturation_pct)
    uncited = check_uncited_canonical_writes(audit_log_path)

    anchor_total = int(conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0])
    model_total = int(conn.execute("SELECT COUNT(*) FROM models").fetchone()[0])

    return CoherenceReport(
        bank_correlations=all_corr,
        bank_correlations_suspicious=suspicious,
        null_coverage_per_bank=nulls,
        anchor_orphans=orphans,
        anchor_oversaturated=oversat,
        anchor_total=anchor_total,
        model_total=model_total,
        uncited_canonical_writes=uncited,
    )


def format_report_human(report: CoherenceReport) -> str:
    """Render the report as a readable Markdown summary."""
    lines: list[str] = []
    lines.append(f"# Coherence audit")
    lines.append("")
    lines.append(
        f"- Models: {report.model_total}  |  Anchors: {report.anchor_total}"
    )
    lines.append("")

    lines.append("## Bank orthogonality")
    if not report.bank_correlations:
        lines.append("(insufficient data)")
    elif not report.bank_correlations_suspicious:
        lines.append("All pairwise |r| below threshold. Banks remain orthogonal.")
    else:
        lines.append("Suspicious pairs (|r| ≥ threshold):")
        for pair, corr in sorted(
            report.bank_correlations_suspicious.items(),
            key=lambda x: -abs(x[1]),
        ):
            lines.append(f"  - {pair}: r = {corr:+.4f}")
    lines.append("")

    lines.append("## NULL coverage per bank")
    for bank, stats in report.null_coverage_per_bank.items():
        lines.append(
            f"  - {bank}: {stats['coverage_pct']:.1f}% covered  "
            f"({stats['models_with']} with, {stats['missing']} missing)"
        )
    lines.append("")

    lines.append("## Anchor orphans (zero assignments)")
    if not report.anchor_orphans:
        lines.append("(none)")
    else:
        lines.append(f"  {len(report.anchor_orphans)} orphans:")
        for label in report.anchor_orphans[:20]:
            lines.append(f"  - {label}")
        if len(report.anchor_orphans) > 20:
            lines.append(f"  ... and {len(report.anchor_orphans) - 20} more")
    lines.append("")

    lines.append("## Anchor oversaturation (>50% of models)")
    if not report.anchor_oversaturated:
        lines.append("(none)")
    else:
        for item in report.anchor_oversaturated:
            lines.append(
                f"  - {item['label']} [{item['bank']}]: "
                f"{item['assignments']} models ({item['share_pct']:.1f}%)"
            )
    lines.append("")

    lines.append("## Uncited canonical writes")
    if not report.uncited_canonical_writes:
        lines.append("(none)")
    else:
        lines.append(f"  {len(report.uncited_canonical_writes)} entries flagged:")
        for item in report.uncited_canonical_writes[:10]:
            issue = item["issue"]
            reason = item["reason"] or "<empty>"
            entry = item["entry"]
            ts = entry.get("ts", "?")
            table = entry.get("table", "?")
            lines.append(f"  - [{issue}] {ts} {table}: reason={reason!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m model_atlas.coherence``."""
    parser = argparse.ArgumentParser(description="Run the coherence audit.")
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of Markdown."
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.7,
        help="Flag bank pairs with |r| at or above this value.",
    )
    parser.add_argument(
        "--oversaturation-pct",
        type=float,
        default=50.0,
        help="Flag anchors assigned to more than this percent of models.",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=None,
        help="Path to the audit log (defaults to data/patches.jsonl).",
    )
    args = parser.parse_args(argv)

    conn = get_connection()
    try:
        report = run_audit(
            conn,
            correlation_threshold=args.correlation_threshold,
            oversaturation_pct=args.oversaturation_pct,
            audit_log_path=args.audit_log,
        )
    finally:
        conn.close()

    if args.json:
        sys.stdout.write(json.dumps(report.to_dict(), indent=2, default=str))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_report_human(report))
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
