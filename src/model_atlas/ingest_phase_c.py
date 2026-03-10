"""Phase C export orchestration and status for ModelAtlas.

Exports JSONL shards for standalone workers. Merge logic lives in
ingest_phase_c_merge.py (re-exported here for backward compatibility).
"""

from __future__ import annotations

import json
import logging
import sqlite3

from .config import (
    PHASE_C1_WORK_DIR,
    PHASE_C3_WORK_DIR,
    PHASE_C_WORK_DIR,
    QUALITY_GATE_MIN_SCORE,
)
from .extraction.vibes import build_quality_gate_prompt, build_vibe_prompt
from .ingest_phase_c_merge import merge_c1 as merge_c1  # noqa: F401
from .ingest_phase_c_merge import merge_c2 as merge_c2  # noqa: F401
from .ingest_phase_c_merge import merge_c3 as merge_c3  # noqa: F401

logger = logging.getLogger(__name__)


# Helpers


def _get_all_anchor_labels(conn: sqlite3.Connection, model_id: str) -> list[str]:
    """All anchor labels for a model."""
    rows = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ?""",
        (model_id,),
    ).fetchall()
    return [r[0] for r in rows]


def _get_anchor_labels_by_bank(conn: sqlite3.Connection, bank: str) -> list[str]:
    """All anchor labels in a bank (dictionary-wide, not per-model)."""
    rows = conn.execute(
        "SELECT label FROM anchors WHERE bank = ? ORDER BY label",
        (bank,),
    ).fetchall()
    return [r[0] for r in rows]


def _get_metadata(conn: sqlite3.Connection, model_id: str, key: str) -> str | None:
    """Single metadata value, or None."""
    row = conn.execute(
        "SELECT value FROM model_metadata WHERE model_id = ? AND key = ?",
        (model_id, key),
    ).fetchone()
    return row[0] if row else None


def _models_without_metadata(
    conn: sqlite3.Connection, key: str, min_likes: int = 0
) -> list[str]:
    """Model IDs that are missing a metadata key.

    Optionally filtered by a minimum likes threshold (stored as metadata).
    """
    if min_likes > 0:
        rows = conn.execute(
            """SELECT m.model_id FROM models m
               LEFT JOIN model_metadata mm
                 ON m.model_id = mm.model_id AND mm.key = ?
               LEFT JOIN model_metadata ml
                 ON m.model_id = ml.model_id AND ml.key = 'likes'
               WHERE mm.value IS NULL
                 AND CAST(COALESCE(ml.value, '0') AS INTEGER) >= ?
               ORDER BY CAST(COALESCE(ml.value, '0') AS INTEGER) DESC""",
            (key, min_likes),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT m.model_id FROM models m
               LEFT JOIN model_metadata mm
                 ON m.model_id = mm.model_id AND mm.key = ?
               WHERE mm.value IS NULL
               ORDER BY m.model_id""",
            (key,),
        ).fetchall()
    return [r[0] for r in rows]


def _get_family(conn: sqlite3.Connection, model_id: str) -> str:
    """Get family anchor label for a model."""
    row = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ? AND a.bank = 'LINEAGE'
             AND a.category = 'family'
           LIMIT 1""",
        (model_id,),
    ).fetchone()
    return row[0] if row else "unknown"


def _get_param_count(conn: sqlite3.Connection, model_id: str) -> str:
    """Get parameter count string from metadata."""
    row = conn.execute(
        "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'parameter_count_b'",
        (model_id,),
    ).fetchone()
    if row:
        return f"{row[0]}B parameters"
    return "unknown"


def _build_config_summary(conn: sqlite3.Connection, model_id: str) -> str:
    """Build a config summary string from metadata fields."""
    keys = [
        "model_type",
        "num_layers",
        "hidden_size",
        "num_heads",
        "vocab_size",
        "context_length",
    ]
    parts = []
    for key in keys:
        val = _get_metadata(conn, model_id, key)
        if val:
            parts.append(f"{key}={val}")
    return ", ".join(parts)


def _get_author(conn: sqlite3.Connection, model_id: str) -> str:
    """Get model author."""
    row = conn.execute(
        "SELECT author FROM models WHERE model_id = ?", (model_id,)
    ).fetchone()
    return row[0] if row else ""


def _get_pipeline_tag(conn: sqlite3.Connection, model_id: str) -> str:
    """Get pipeline_tag from metadata."""
    val = _get_metadata(conn, model_id, "pipeline_tag")
    return val or ""


# Export C1


def export_c1(conn: sqlite3.Connection) -> int:
    """Export model_ids for C1 (Smol-Hub-tldr) processing.

    Writes model IDs that don't yet have smol_summary metadata to
    PHASE_C1_WORK_DIR/models_for_c1.jsonl.
    """
    model_ids = _models_without_metadata(conn, "smol_summary")
    if not model_ids:
        logger.info("export_c1: all models already have smol_summary")
        return 0

    PHASE_C1_WORK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PHASE_C1_WORK_DIR / "models_for_c1.jsonl"

    with open(out_path, "w") as f:
        for mid in model_ids:
            f.write(json.dumps({"model_id": mid}) + "\n")

    logger.info("export_c1: wrote %d model_ids to %s", len(model_ids), out_path)
    return len(model_ids)


# Export C2


def _build_c2_prompt(
    conn: sqlite3.Connection,
    mid: str,
    cap_labels: list[str],
    dom_labels: list[str],
) -> str:
    """Build a single C2 vibe prompt for one model."""
    tags = _get_all_anchor_labels(conn, mid)
    existing_set = set(tags)
    return build_vibe_prompt(
        model_id=mid,
        author=_get_author(conn, mid),
        pipeline_tag=_get_pipeline_tag(conn, mid),
        tags=tags,
        param_count=_get_param_count(conn, mid),
        family=_get_family(conn, mid),
        existing_anchors=tags,
        config_summary=_build_config_summary(conn, mid),
        card_excerpt=_get_metadata(conn, mid, "smol_summary") or "",
        capability_candidates=[lb for lb in cap_labels if lb not in existing_set],
        domain_candidates=[lb for lb in dom_labels if lb not in existing_set],
    )


def export_c2(
    conn: sqlite3.Connection,
    num_shards: int = 4,
    min_likes: int = 0,
) -> int:
    """Export C2 vibe prompts to sharded JSONL files."""
    from contextlib import ExitStack

    model_ids = _models_without_metadata(conn, "qwen_summary", min_likes)
    if not model_ids:
        logger.info("export_c2: all models already have qwen_summary")
        return 0

    PHASE_C_WORK_DIR.mkdir(parents=True, exist_ok=True)
    cap_labels = _get_anchor_labels_by_bank(conn, "CAPABILITY")
    dom_labels = _get_anchor_labels_by_bank(conn, "DOMAIN")
    all_valid = sorted(set(cap_labels + dom_labels))

    with ExitStack() as stack:
        shard_files = [
            stack.enter_context(open(PHASE_C_WORK_DIR / f"shard_{i}.jsonl", "w"))
            for i in range(num_shards)
        ]
        for idx, mid in enumerate(model_ids):
            prompt = _build_c2_prompt(conn, mid, cap_labels, dom_labels)
            item = {"model_id": mid, "prompt": prompt, "valid_anchors": all_valid}
            shard_files[idx % num_shards].write(json.dumps(item) + "\n")

    logger.info(
        "export_c2: wrote %d prompts across %d shards", len(model_ids), num_shards
    )
    return len(model_ids)


# Export C3


def export_c3(conn: sqlite3.Connection, num_shards: int = 4) -> int:
    """Export C3 quality gate prompts to sharded JSONL files."""
    from contextlib import ExitStack

    rows = conn.execute(
        """SELECT vs.model_id, vs.value as summary
           FROM model_metadata vs
           LEFT JOIN model_metadata qs
             ON vs.model_id = qs.model_id AND qs.key = 'quality_score'
           WHERE vs.key = 'vibe_summary' AND qs.value IS NULL
           ORDER BY vs.model_id"""
    ).fetchall()
    if not rows:
        logger.info("export_c3: all vibed models already have quality_score")
        return 0

    PHASE_C3_WORK_DIR.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        shard_files = [
            stack.enter_context(open(PHASE_C3_WORK_DIR / f"shard_{i}.jsonl", "w"))
            for i in range(num_shards)
        ]
        for idx, (model_id, summary) in enumerate(rows):
            tags = _get_all_anchor_labels(conn, model_id)
            prompt = build_quality_gate_prompt(
                model_id=model_id, summary=summary, tags=tags
            )
            shard_files[idx % num_shards].write(
                json.dumps({"model_id": model_id, "prompt": prompt}) + "\n"
            )

    logger.info("export_c3: wrote %d prompts across %d shards", len(rows), num_shards)
    return len(rows)


# Select summaries


def select_summaries(conn: sqlite3.Connection) -> dict[str, int]:
    """Pick the best summary per model and store as vibe_summary.

    Priority: smol_summary > qwen_summary.
    Stores vibe_summary + vibe_summary_source metadata.
    Skips models that already have vibe_summary.
    Returns {"selected": N, "smol": N, "qwen": N, "skipped": N}.
    """
    from . import db

    # Get all models in the DB
    rows = conn.execute("SELECT model_id FROM models").fetchall()

    selected = 0
    smol_count = 0
    qwen_count = 0
    skipped = 0

    for row in rows:
        mid = row[0]

        # Skip if already has vibe_summary
        existing = _get_metadata(conn, mid, "vibe_summary")
        if existing:
            skipped += 1
            continue

        # Try smol_summary first, then qwen_summary
        smol = _get_metadata(conn, mid, "smol_summary")
        qwen = _get_metadata(conn, mid, "qwen_summary")

        if smol:
            db.set_metadata(conn, mid, "vibe_summary", smol, "str")
            db.set_metadata(conn, mid, "vibe_summary_source", "smol", "str")
            smol_count += 1
            selected += 1
        elif qwen:
            db.set_metadata(conn, mid, "vibe_summary", qwen, "str")
            db.set_metadata(conn, mid, "vibe_summary_source", "qwen", "str")
            qwen_count += 1
            selected += 1

    conn.commit()
    logger.info(
        "select_summaries: selected=%d (smol=%d qwen=%d) skipped=%d",
        selected,
        smol_count,
        qwen_count,
        skipped,
    )
    return {
        "selected": selected,
        "smol": smol_count,
        "qwen": qwen_count,
        "skipped": skipped,
    }


# Status


def get_phase_c_status(conn: sqlite3.Connection) -> dict:
    """Get Phase C progress from network DB metadata keys."""
    total = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]

    def _count_key(key: str) -> int:
        row = conn.execute(
            "SELECT COUNT(*) FROM model_metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0]

    def _count_passing(min_score: float) -> int:
        row = conn.execute(
            "SELECT COUNT(*) FROM model_metadata WHERE key = 'quality_score' AND CAST(value AS REAL) >= ?",
            (min_score,),
        ).fetchone()
        return row[0]

    smol = _count_key("smol_summary")
    qwen = _count_key("qwen_summary")
    quality = _count_key("quality_score")
    vibe = _count_key("vibe_summary")
    passing = _count_passing(QUALITY_GATE_MIN_SCORE)

    return {
        "total_models": total,
        "smol_summary": smol,
        "qwen_summary": qwen,
        "quality_score": quality,
        "quality_passing": passing,
        "quality_failing": quality - passing,
        "vibe_summary": vibe,
    }


def print_phase_c_status(conn: sqlite3.Connection) -> None:
    """Print human-readable Phase C status."""
    status = get_phase_c_status(conn)
    print("\nPhase C Status (network DB)")
    print(f"{'=' * 40}")
    print(f"Total models:         {status['total_models']}")
    print(f"C1 smol_summary:      {status['smol_summary']}")
    print(f"C2 qwen_summary:      {status['qwen_summary']}")
    print(
        f"C3 quality_score:     {status['quality_score']} (pass={status['quality_passing']} fail={status['quality_failing']})"
    )
    print(f"Selected vibe_summary:{status['vibe_summary']}")
