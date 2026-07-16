#!/usr/bin/env python3
"""One-shot scrub for multi-tier anchor collisions.

Two dirt patterns removed here:
  1. Multiple long-context-* tiers on one model — keep the tightest bound only
     (32k is subsumed by 128k which is subsumed by 1m). Matches the invariant
     the fixed _context_length_anchors now emits.
  2. Multiple *B-class size anchors on one model — if the model has a known
     parameter_count_b metadata, keep the size-class that BEST fits; otherwise
     keep the largest class (conservative — matches phase_e_postprocess
     Layer 1's fallback behaviour).

Dry-run by default. --apply to actually delete.
Snapshot the DB before running with --apply.
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict

from model_atlas import db

# Ordered smallest -> largest — matches _context_length_anchors emission
LONG_CTX_ORDER = ["long-context-32k", "long-context-128k", "long-context-1m"]

# (min_b_inclusive, max_b_exclusive, label) — matches _PARAM_RANGES in
# extraction/deterministic.py. Duplicated here rather than imported so the
# scrub is self-contained if ranges change later.
SIZE_CLASSES = [
    (0.0, 0.5, "sub-1B"),
    (0.5, 1.5, "1B-class"),
    (1.5, 5.0, "3B-class"),
    (5.0, 10.0, "7B-class"),
    (10.0, 20.0, "13B-class"),
    (20.0, 50.0, "30B-class"),
    (50.0, 100.0, "70B-class"),
    (100.0, float("inf"), "frontier-class"),
]
SIZE_LABELS = [row[2] for row in SIZE_CLASSES]


def _load_anchor_labels(conn: sqlite3.Connection) -> dict[str, int]:
    return {r[0]: r[1] for r in conn.execute("SELECT label, anchor_id FROM anchors")}


def _load_param_counts(conn: sqlite3.Connection) -> dict[str, float]:
    """model_id -> param_b (from parameter_count_b metadata, when present)."""
    out: dict[str, float] = {}
    for mid, val in conn.execute(
        "SELECT model_id, value FROM model_metadata WHERE key = 'parameter_count_b'"
    ):
        try:
            out[mid] = float(val)
        except (TypeError, ValueError):
            pass
    return out


def _scrub_long_context(
    conn: sqlite3.Connection, anchor_ids: dict[str, int]
) -> tuple[list[tuple[str, int]], int]:
    """Return (rows_to_delete, models_normalized) for long-context multi-tiers."""
    labels_present = [L for L in LONG_CTX_ORDER if L in anchor_ids]
    ids_of = {L: anchor_ids[L] for L in labels_present}
    ids_set = set(ids_of.values())
    if not ids_set:
        return [], 0

    per_model: dict[str, set[int]] = defaultdict(set)
    placeholders = ",".join("?" * len(ids_set))
    for mid, aid in conn.execute(
        f"SELECT model_id, anchor_id FROM model_anchors WHERE anchor_id IN ({placeholders})",
        tuple(ids_set),
    ):
        per_model[mid].add(aid)

    id_to_rank = {ids_of[L]: rank for rank, L in enumerate(labels_present)}
    to_delete: list[tuple[str, int]] = []
    normalized = 0
    for mid, aids in per_model.items():
        if len(aids) < 2:
            continue
        keeper = max(aids, key=lambda a: id_to_rank[a])
        for aid in aids:
            if aid != keeper:
                to_delete.append((mid, aid))
        normalized += 1
    return to_delete, normalized


def _best_size_label(param_b: float) -> str | None:
    for lo, hi, label in SIZE_CLASSES:
        if lo <= param_b < hi:
            return label
    return None


def _scrub_size_class(
    conn: sqlite3.Connection,
    anchor_ids: dict[str, int],
    param_counts: dict[str, float],
) -> tuple[list[tuple[str, int]], int, int]:
    """Return (rows_to_delete, normalized_by_param, normalized_by_fallback)."""
    ids_of = {L: anchor_ids[L] for L in SIZE_LABELS if L in anchor_ids}
    label_of = {v: k for k, v in ids_of.items()}
    ids_set = set(ids_of.values())
    if not ids_set:
        return [], 0, 0

    per_model: dict[str, set[int]] = defaultdict(set)
    placeholders = ",".join("?" * len(ids_set))
    for mid, aid in conn.execute(
        f"SELECT model_id, anchor_id FROM model_anchors WHERE anchor_id IN ({placeholders})",
        tuple(ids_set),
    ):
        per_model[mid].add(aid)

    # rank labels by size (largest last)
    label_rank = {label: i for i, label in enumerate(SIZE_LABELS)}
    to_delete: list[tuple[str, int]] = []
    by_param = 0
    by_fallback = 0
    for mid, aids in per_model.items():
        if len(aids) < 2:
            continue
        param_b = param_counts.get(mid)
        keeper = None
        if param_b is not None:
            best = _best_size_label(param_b)
            if best and best in ids_of and ids_of[best] in aids:
                keeper = ids_of[best]
        if keeper is None:
            # Fallback: keep the LARGEST class present
            keeper = max(aids, key=lambda a: label_rank[label_of[a]])
            by_fallback += 1
        else:
            by_param += 1
        for aid in aids:
            if aid != keeper:
                to_delete.append((mid, aid))
    return to_delete, by_param, by_fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete rows (default: dry-run).")
    args = parser.parse_args()

    conn = db.get_connection()
    db.init_db(conn)

    anchor_ids = _load_anchor_labels(conn)
    param_counts = _load_param_counts(conn)

    lc_deletes, lc_models = _scrub_long_context(conn, anchor_ids)
    sc_deletes, sc_by_param, sc_by_fallback = _scrub_size_class(conn, anchor_ids, param_counts)

    print("=== long-context tier collisions ===")
    print(f"  models normalized: {lc_models}")
    print(f"  rows to delete:    {len(lc_deletes)}")

    print()
    print("=== size-class tier collisions ===")
    print(f"  normalized via param_count_b: {sc_by_param}")
    print(f"  normalized via largest-fallback: {sc_by_fallback}")
    print(f"  rows to delete: {len(sc_deletes)}")

    total = len(lc_deletes) + len(sc_deletes)
    print()
    print(f"TOTAL rows to delete: {total}")

    if not args.apply:
        print("(dry-run — pass --apply to execute)")
        return 0

    if total == 0:
        print("(nothing to do)")
        return 0

    cur = conn.cursor()
    for mid, aid in lc_deletes + sc_deletes:
        cur.execute(
            "DELETE FROM model_anchors WHERE model_id = ? AND anchor_id = ?",
            (mid, aid),
        )
    conn.commit()
    print(f"applied — deleted {cur.rowcount} rows in final DELETE batch (cumulative: {total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
