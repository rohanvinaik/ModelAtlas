"""Phase C merge: apply worker results to the network DB.

Shared JSONL parsing and per-phase merge logic for C1/C2/C3 results.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from . import db
from .config import QUALITY_GATE_MIN_SCORE

logger = logging.getLogger(__name__)


def _parse_jsonl_line(line: str, required_field: str) -> tuple[dict | None, str]:
    """Parse one JSONL line with standard validation.

    Returns (item, status) where status is "ok", "empty", "error", or "skip".
    """
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
    if not item.get(required_field):
        return None, "skip"
    return item, "ok"


def _iter_jsonl_items(files: list[str], required_field: str):
    """Yield (item, status) from JSONL files, skipping empty lines."""
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                item, status = _parse_jsonl_line(line, required_field)
                if status != "empty":
                    yield item, status


def _merge_c1_item(conn: sqlite3.Connection, item: dict) -> None:
    """Apply one C1 merge item: ensure model exists, store smol_summary."""
    model_id = item["model_id"]
    existing = conn.execute(
        "SELECT 1 FROM models WHERE model_id = ?", (model_id,)
    ).fetchone()
    if not existing:
        db.insert_model(conn, model_id, source="stub")
    db.set_metadata(conn, model_id, "smol_summary", item["smol_summary"], "str")


def merge_c1(conn: sqlite3.Connection, files: list[str]) -> dict[str, int]:
    """Merge C1 Smol-Hub-tldr results into network DB.

    Reads JSONL with {"model_id", "smol_summary"} records.
    Creates stub models for unknown model_ids.
    Returns {"merged": N, "skipped": N, "errors": N}.
    """
    merged = skipped = errors = 0
    for item, status in _iter_jsonl_items(files, "smol_summary"):
        if status == "error":
            errors += 1
        elif status == "skip":
            skipped += 1
        else:
            _merge_c1_item(conn, item)
            merged += 1

    conn.commit()
    logger.info("merge_c1: merged=%d skipped=%d errors=%d", merged, skipped, errors)
    return {"merged": merged, "skipped": skipped, "errors": errors}


def _merge_c2_item(conn: sqlite3.Connection, item: dict) -> int:
    """Apply one C2 merge item: store summary + link anchors. Returns anchors linked."""
    model_id = item["model_id"]
    db.set_metadata(conn, model_id, "qwen_summary", item["summary"], "str")

    selected = item.get("selected_anchors") or item.get("extra_anchors") or []
    linked = 0
    for anchor_label in selected[:5]:
        anchor_label = anchor_label.strip().lower()
        if not anchor_label:
            continue
        row = conn.execute(
            "SELECT anchor_id FROM anchors WHERE label = ?",
            (anchor_label,),
        ).fetchone()
        if row:
            db.link_anchor(conn, model_id, row[0], confidence=0.5)
            linked += 1
    return linked


def merge_c2(conn: sqlite3.Connection, files: list[str]) -> dict[str, int]:
    """Merge C2 vibe results into network DB.

    Reads JSONL with {"model_id", "summary", "selected_anchors"} records.
    Stores qwen_summary metadata + links existing dictionary anchors
    (confidence=0.5). Only anchors already in the DB are linked.
    Returns {"merged": N, "skipped": N, "errors": N, "anchors_linked": N}.
    """
    merged = skipped = errors = anchors_linked = 0
    for item, status in _iter_jsonl_items(files, "summary"):
        if status == "error":
            errors += 1
        elif status == "skip":
            skipped += 1
        else:
            anchors_linked += _merge_c2_item(conn, item)
            merged += 1

    conn.commit()
    logger.info(
        "merge_c2: merged=%d skipped=%d errors=%d anchors_linked=%d",
        merged, skipped, errors, anchors_linked,
    )
    return {
        "merged": merged,
        "skipped": skipped,
        "errors": errors,
        "anchors_linked": anchors_linked,
    }


def _merge_c3_item(conn: sqlite3.Connection, item: dict) -> bool:
    """Apply one C3 merge item: store scores + flags. Returns True if passed quality gate."""
    model_id = item["model_id"]
    quality_score = item["quality_score"]

    db.set_metadata(conn, model_id, "quality_score", str(quality_score), "float")

    for sub_key in ("specificity", "coherence", "artifacts"):
        val = item.get(sub_key)
        if val is not None:
            db.set_metadata(conn, model_id, f"quality_{sub_key}", str(val), "int")

    flags = item.get("flags", [])
    if flags:
        db.set_metadata(conn, model_id, "quality_flags", json.dumps(flags), "json")

    return quality_score >= QUALITY_GATE_MIN_SCORE


def merge_c3(conn: sqlite3.Connection, files: list[str]) -> dict[str, int]:
    """Merge C3 quality gate results into network DB.

    Reads JSONL with {"model_id", "quality_score", "specificity",
    "coherence", "artifacts", "flags"} records.
    Returns {"merged": N, "skipped": N, "errors": N, "passed": N, "failed": N}.
    """
    merged = skipped = errors = passed = failed = 0
    for item, status in _iter_jsonl_items(files, "quality_score"):
        if status == "error":
            errors += 1
        elif status == "skip":
            skipped += 1
        else:
            if _merge_c3_item(conn, item):
                passed += 1
            else:
                failed += 1
            merged += 1

    conn.commit()
    logger.info(
        "merge_c3: merged=%d (passed=%d failed=%d) skipped=%d errors=%d",
        merged, passed, failed, skipped, errors,
    )
    return {
        "merged": merged,
        "skipped": skipped,
        "errors": errors,
        "passed": passed,
        "failed": failed,
    }
