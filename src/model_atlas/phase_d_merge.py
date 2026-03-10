"""D3 merge: apply healing results to the network DB.

Reads JSONL heal result files, computes anchor diffs, applies changes,
and stores correction_events for DPO training data.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from . import db

logger = logging.getLogger(__name__)


def _compute_anchor_diff(
    item: dict,
) -> tuple[dict, set[str], list[str], list[str]]:
    """Parse original response and compute anchor diff.

    Returns (original_dict, healed_anchors_set, anchors_added, anchors_removed).
    """
    original_response_str = item.get("original_response", "{}")
    try:
        original = json.loads(original_response_str)
    except (json.JSONDecodeError, TypeError):
        original = {}
    original_anchors = set(original.get("selected_anchors", []))

    selected_anchors = item.get("selected_anchors", [])
    healed_anchors = {
        a.strip().lower() for a in selected_anchors if isinstance(a, str) and a.strip()
    }

    anchors_added = sorted(healed_anchors - original_anchors)
    anchors_removed = sorted(original_anchors - healed_anchors)
    return original, healed_anchors, anchors_added, anchors_removed


def _apply_anchor_changes(
    conn: sqlite3.Connection,
    model_id: str,
    anchors_added: list[str],
    anchors_removed: list[str],
) -> tuple[int, int]:
    """Apply anchor add/remove operations. Returns (added_count, removed_count)."""
    added = 0
    for label in anchors_added:
        row = conn.execute(
            "SELECT anchor_id FROM anchors WHERE label = ?",
            (label,),
        ).fetchone()
        if row:
            db.link_anchor(conn, model_id, row[0], confidence=0.6)
            added += 1

    for label in anchors_removed:
        conn.execute(
            """DELETE FROM model_anchors
               WHERE model_id = ?
                 AND anchor_id = (SELECT anchor_id FROM anchors WHERE label = ?)
                 AND confidence = 0.5""",
            (model_id, label),
        )

    return added, len(anchors_removed)


def _merge_single_item(
    conn: sqlite3.Connection,
    item: dict,
    run_id: str,
) -> tuple[int, int]:
    """Merge one healed item: apply changes, store correction. Returns (added, removed)."""
    model_id = item["model_id"]
    original, healed_anchors, anchors_added, anchors_removed = _compute_anchor_diff(
        item
    )
    added, removed = _apply_anchor_changes(
        conn, model_id, anchors_added, anchors_removed
    )

    summary = item["summary"]
    if summary != original.get("summary", ""):
        db.set_metadata(conn, model_id, "qwen_summary", summary, "str")

    healed_response = json.dumps(
        {
            "summary": summary,
            "selected_anchors": sorted(healed_anchors),
        }
    )
    db.insert_correction_event(
        conn,
        run_id=run_id,
        model_id=model_id,
        tier=item.get("tier", "local"),
        original_prompt=item.get("original_prompt"),
        original_response=item.get("original_response", "{}"),
        healed_response=healed_response,
        anchors_added=anchors_added,
        anchors_removed=anchors_removed,
        rationale=item.get("rationale", ""),
    )
    return added, removed


def _parse_heal_line(line: str) -> tuple[dict | None, str]:
    """Parse one JSONL heal result line. Returns (item, status)."""
    line = line.strip()
    if not line:
        return None, "empty"
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        return None, "error"
    if not item.get("model_id"):
        return None, "error"
    if "error" in item or not item.get("summary"):
        return None, "skip"
    return item, "ok"


def _iter_heal_items(files: list[str]):
    """Yield (item_dict, status) tuples from JSONL heal result files."""
    for fpath in files:
        resolved = Path(fpath).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"JSONL file not found: {fpath}")
        with open(resolved) as f:
            for line in f:
                item, status = _parse_heal_line(line)
                if status != "empty":
                    yield item, status


def merge_d3(
    conn: sqlite3.Connection,
    files: list[str],
    run_id: str,
) -> dict[str, int]:
    """Merge D3 healing results into network DB.

    Returns {"merged": N, "skipped": N, "errors": N, "anchors_added": N, "anchors_removed": N}.
    """
    merged = skipped = errors = total_added = total_removed = 0

    for item, status in _iter_heal_items(files):
        if status == "error":
            errors += 1
        elif status == "skip":
            skipped += 1
        else:
            added, removed = _merge_single_item(conn, item, run_id)
            total_added += added
            total_removed += removed
            merged += 1

    conn.commit()
    result = {
        "merged": merged,
        "skipped": skipped,
        "errors": errors,
        "anchors_added": total_added,
        "anchors_removed": total_removed,
    }
    db.finish_phase_d_run(conn, run_id, "completed", result)
    conn.commit()

    logger.info(
        "merge_d3: merged=%d skipped=%d errors=%d added=%d removed=%d",
        merged,
        skipped,
        errors,
        total_added,
        total_removed,
    )
    return result
