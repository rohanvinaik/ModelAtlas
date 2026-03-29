"""Phase E merge: apply web enrichment results to the network DB.

QC validation before touching existing data, authority-weighted blending
(confidence 0.4), dry-run mode, and shadow ledger logging via phase_d_runs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from . import db

logger = logging.getLogger(__name__)

# Web extraction sits below all existing sources
WEB_EXTRACTION_CONFIDENCE = 0.4


def _parse_jsonl_line(line: str) -> tuple[dict | None, str]:
    """Parse one JSONL line. Returns (item, status)."""
    line = line.strip()
    if not line:
        return None, "empty"
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        return None, "error"
    if not item.get("model_id"):
        return None, "error"
    if "error" in item:
        return None, "skip"
    return item, "ok"


def _iter_jsonl_items(files: list[str]):
    """Yield (item, status) from JSONL files."""
    for fpath in files:
        resolved = Path(fpath).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"JSONL file not found: {fpath}")
        with open(resolved) as f:
            for line in f:
                item, status = _parse_jsonl_line(line)
                if status != "empty":
                    yield item, status


def _get_existing_anchor_confidence(
    conn: sqlite3.Connection, model_id: str, anchor_id: int
) -> float | None:
    """Get current confidence for a model-anchor link, or None if unlinked."""
    row = conn.execute(
        "SELECT confidence FROM model_anchors WHERE model_id = ? AND anchor_id = ?",
        (model_id, anchor_id),
    ).fetchone()
    return row[0] if row else None


def _validate_anchor(conn: sqlite3.Connection, label: str) -> int | None:
    """Look up anchor_id by label. Returns None if not in dictionary."""
    row = conn.execute(
        "SELECT anchor_id FROM anchors WHERE label = ?", (label,)
    ).fetchone()
    return row[0] if row else None


def _qc_check_contradiction(
    conn: sqlite3.Connection, model_id: str, anchor_id: int
) -> bool:
    """Check if adding this anchor contradicts high-confidence existing data.

    Returns True if the anchor should be SKIPPED (contradiction found).
    """
    existing_conf = _get_existing_anchor_confidence(conn, model_id, anchor_id)
    if existing_conf is not None and existing_conf >= 0.85:
        # Already linked at high confidence — skip, don't overwrite
        return True
    return False


def _link_bank_anchors(
    conn: sqlite3.Connection,
    model_id: str,
    selected: list,
    dry_run: bool,
    stats: dict,
) -> list[str]:
    """Validate and link anchors for one bank. Returns list of newly linked labels."""
    new_anchors: list[str] = []
    for label in selected:
        label = label.strip().lower()
        anchor_id = _validate_anchor(conn, label)
        if anchor_id is None:
            stats["anchors_skipped_invalid"] += 1
            continue

        existing_conf = _get_existing_anchor_confidence(conn, model_id, anchor_id)
        if existing_conf is not None and existing_conf >= WEB_EXTRACTION_CONFIDENCE:
            stats["anchors_skipped_existing"] += 1
            continue

        if _qc_check_contradiction(conn, model_id, anchor_id):
            stats["anchors_skipped_contradiction"] += 1
            continue

        if not dry_run:
            db.link_anchor(conn, model_id, anchor_id, confidence=WEB_EXTRACTION_CONFIDENCE)
        stats["anchors_linked"] += 1
        new_anchors.append(label)
    return new_anchors


def _store_benchmarks(
    conn: sqlite3.Connection,
    model_id: str,
    benchmarks: dict,
    dry_run: bool,
    stats: dict,
) -> None:
    """Validate and store benchmark scores."""
    for bench_name, bench_val in benchmarks.items():
        if isinstance(bench_val, (int, float)) and 0 <= bench_val <= 100:
            if not dry_run:
                db.set_metadata(conn, model_id, f"benchmark:{bench_name}", str(bench_val), "float")
            stats["benchmarks_stored"] += 1


def _merge_one_item(
    conn: sqlite3.Connection, item: dict, dry_run: bool = False
) -> dict:
    """Merge one Phase E result. Returns per-item stats."""
    model_id = item["model_id"]
    banks = item.get("banks") or {}
    source_urls = item.get("source_urls") or []
    web_summary = item.get("web_summary") or ""

    stats = {
        "anchors_linked": 0,
        "anchors_skipped_existing": 0,
        "anchors_skipped_invalid": 0,
        "anchors_skipped_contradiction": 0,
        "benchmarks_stored": 0,
    }

    exists = conn.execute(
        "SELECT 1 FROM models WHERE model_id = ?", (model_id,)
    ).fetchone()
    if not exists:
        return stats

    all_new_anchors: list[str] = []
    for bank_result in banks.values():
        selected = bank_result.get("selected_anchors") or []
        benchmarks = bank_result.get("benchmark_scores") or {}

        new = _link_bank_anchors(conn, model_id, selected, dry_run, stats)
        all_new_anchors.extend(new)
        _store_benchmarks(conn, model_id, benchmarks, dry_run, stats)

    if not dry_run and (all_new_anchors or web_summary):
        if web_summary:
            db.set_metadata(conn, model_id, "web_summary", web_summary, "str")
        if source_urls:
            db.set_metadata(conn, model_id, "web_sources", json.dumps(source_urls[:5]), "json")
        db.set_metadata(conn, model_id, "web_enriched", "true", "str")

    return stats


def merge_phase_e(
    conn: sqlite3.Connection,
    files: list[str],
    dry_run: bool = False,
) -> dict:
    """Merge Phase E web enrichment results into network DB.

    QC validation ensures web extractions don't overwrite higher-confidence
    data. Dry-run mode computes all changes without writing.

    Returns summary statistics.
    """
    run_id = None
    if not dry_run:
        run_id = db.create_phase_d_run(
            conn,
            phase="e1",
            config={"files": [str(Path(f).name) for f in files]},
        )
        conn.commit()

    merged = 0
    skipped = 0
    errors = 0
    total_stats = {
        "anchors_linked": 0,
        "anchors_skipped_existing": 0,
        "anchors_skipped_invalid": 0,
        "anchors_skipped_contradiction": 0,
        "benchmarks_stored": 0,
    }

    for item, status in _iter_jsonl_items(files):
        if status == "error":
            errors += 1
            continue
        if status == "skip" or item is None:
            skipped += 1
            continue

        item_stats = _merge_one_item(conn, item, dry_run=dry_run)
        for k, v in item_stats.items():
            total_stats[k] += v
        merged += 1

    if not dry_run:
        conn.commit()
        if run_id:
            summary = {
                "merged": merged,
                "skipped": skipped,
                "errors": errors,
                **total_stats,
            }
            db.finish_phase_d_run(conn, run_id, "completed", summary)
            conn.commit()

    result = {
        "merged": merged,
        "skipped": skipped,
        "errors": errors,
        "dry_run": dry_run,
        **total_stats,
    }

    action = "DRY RUN" if dry_run else "merge"
    logger.info(
        "merge_phase_e %s: merged=%d skipped=%d errors=%d "
        "anchors_linked=%d benchmarks=%d",
        action,
        merged,
        skipped,
        errors,
        total_stats["anchors_linked"],
        total_stats["benchmarks_stored"],
    )
    return result


def phase_e_status(conn: sqlite3.Connection) -> dict:
    """Get Phase E progress statistics."""
    web_enriched = conn.execute(
        "SELECT COUNT(*) FROM model_metadata WHERE key = 'web_enriched'"
    ).fetchone()[0]

    web_summaries = conn.execute(
        "SELECT COUNT(*) FROM model_metadata WHERE key = 'web_summary'"
    ).fetchone()[0]

    benchmarks = conn.execute(
        "SELECT COUNT(*) FROM model_metadata WHERE key LIKE 'benchmark:%'"
    ).fetchone()[0]

    benchmark_models = conn.execute(
        "SELECT COUNT(DISTINCT model_id) FROM model_metadata WHERE key LIKE 'benchmark:%'"
    ).fetchone()[0]

    total_models = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]

    # Recent Phase E runs
    runs = conn.execute(
        """SELECT run_id, started_at, status, summary
           FROM phase_d_runs WHERE phase LIKE 'e%'
           ORDER BY started_at DESC LIMIT 5"""
    ).fetchall()

    # Web-sourced anchor links (confidence = 0.4)
    web_anchors = conn.execute(
        "SELECT COUNT(*) FROM model_anchors WHERE confidence = ?",
        (WEB_EXTRACTION_CONFIDENCE,),
    ).fetchone()[0]

    return {
        "web_enriched_models": web_enriched,
        "total_models": total_models,
        "coverage_pct": round(web_enriched / max(total_models, 1) * 100, 1),
        "web_summaries": web_summaries,
        "web_anchor_links": web_anchors,
        "benchmark_scores": benchmarks,
        "benchmark_models": benchmark_models,
        "recent_runs": [
            {
                "run_id": r[0][:8],
                "started": r[1],
                "status": r[2],
                "summary": json.loads(r[3]) if r[3] else None,
            }
            for r in runs
        ],
    }
