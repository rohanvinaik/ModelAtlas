"""Phase C export/merge orchestration for ModelAtlas.

Exports JSONL shards for standalone workers and merges results back into
the network DB. State is tracked via metadata keys in network.db:
  - smol_summary  → C1 done
  - qwen_summary  → C2 done
  - quality_score  → C3 done
  - vibe_summary   → summary selected

Functions:
  export_c1   — model_ids for Smol-Hub-tldr worker
  merge_c1    — merge C1 smol_summary results
  export_c2   — vibe prompts sharded for Ollama worker
  merge_c2    — merge C2 qwen_summary + extra anchors
  export_c3   — quality gate prompts sharded for C3 worker
  merge_c3    — merge C3 quality scores
  select_summaries — pick best summary per model
  get_phase_c_status / print_phase_c_status — progress display
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_all_anchor_labels(conn: sqlite3.Connection, model_id: str) -> list[str]:
    """All anchor labels for a model."""
    rows = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ?""",
        (model_id,),
    ).fetchall()
    return [r[0] for r in rows]


def _get_anchors_by_bank(
    conn: sqlite3.Connection, model_id: str, bank: str
) -> list[str]:
    """Anchor labels for a model filtered by bank."""
    rows = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ? AND a.bank = ?""",
        (model_id, bank),
    ).fetchall()
    return [r[0] for r in rows]


def _get_anchor_labels_by_bank(conn: sqlite3.Connection, bank: str) -> list[str]:
    """All anchor labels in a bank (dictionary-wide, not per-model)."""
    rows = conn.execute(
        "SELECT label FROM anchors WHERE bank = ? ORDER BY label",
        (bank,),
    ).fetchall()
    return [r[0] for r in rows]


def _get_metadata(
    conn: sqlite3.Connection, model_id: str, key: str
) -> str | None:
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


def _get_training_method(conn: sqlite3.Connection, model_id: str) -> str:
    """Get training method anchor label for a model."""
    rows = _get_anchors_by_bank(conn, model_id, "TRAINING")
    return rows[0] if rows else "unknown"


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
    keys = ["model_type", "num_layers", "hidden_size", "num_heads", "vocab_size", "context_length"]
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


# ---------------------------------------------------------------------------
# Export C1
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Merge C1
# ---------------------------------------------------------------------------


def merge_c1(conn: sqlite3.Connection, files: list[str]) -> dict[str, int]:
    """Merge C1 Smol-Hub-tldr results into network DB.

    Reads JSONL with {"model_id", "smol_summary"} records.
    Creates stub models for unknown model_ids.
    Returns {"merged": N, "skipped": N, "errors": N}.
    """
    from . import db

    merged = 0
    skipped = 0
    errors = 0

    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                model_id = item.get("model_id", "")
                if not model_id:
                    errors += 1
                    continue

                if "error" in item:
                    skipped += 1
                    continue

                summary = item.get("smol_summary", "")
                if not summary:
                    skipped += 1
                    continue

                # Ensure model exists (create stub if not)
                existing = conn.execute(
                    "SELECT 1 FROM models WHERE model_id = ?", (model_id,)
                ).fetchone()
                if not existing:
                    db.insert_model(conn, model_id, source="stub")

                db.set_metadata(conn, model_id, "smol_summary", summary, "str")
                merged += 1

    conn.commit()
    logger.info(
        "merge_c1: merged=%d skipped=%d errors=%d", merged, skipped, errors
    )
    return {"merged": merged, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Export C2
# ---------------------------------------------------------------------------


def export_c2(
    conn: sqlite3.Connection,
    num_shards: int = 4,
    min_likes: int = 0,
) -> int:
    """Export C2 vibe prompts to sharded JSONL files.

    Builds selection prompts from network DB data. The worker selects
    anchors from curated CAPABILITY/DOMAIN dictionary lists rather than
    generating free-form tags. Each JSONL item includes valid_anchors
    for worker-side validation.
    """
    model_ids = _models_without_metadata(conn, "qwen_summary", min_likes)
    if not model_ids:
        logger.info("export_c2: all models already have qwen_summary")
        return 0

    PHASE_C_WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Cache dictionary labels once for the batch
    all_capability_labels = _get_anchor_labels_by_bank(conn, "CAPABILITY")
    all_domain_labels = _get_anchor_labels_by_bank(conn, "DOMAIN")
    all_valid = sorted(set(all_capability_labels + all_domain_labels))

    # Open shard files
    shard_files = []
    for i in range(num_shards):
        shard_files.append(open(PHASE_C_WORK_DIR / f"shard_{i}.jsonl", "w"))

    try:
        for idx, mid in enumerate(model_ids):
            author = _get_author(conn, mid)
            pipeline_tag = _get_pipeline_tag(conn, mid)
            tags = _get_all_anchor_labels(conn, mid)
            param_count = _get_param_count(conn, mid)
            family = _get_family(conn, mid)
            existing_anchors = tags  # all anchors already linked
            config_summary = _build_config_summary(conn, mid)
            card_excerpt = _get_metadata(conn, mid, "smol_summary") or ""

            # Candidates = dictionary labels minus already-assigned
            existing_set = set(existing_anchors)
            cap_candidates = [lb for lb in all_capability_labels if lb not in existing_set]
            dom_candidates = [lb for lb in all_domain_labels if lb not in existing_set]

            prompt = build_vibe_prompt(
                model_id=mid,
                author=author,
                pipeline_tag=pipeline_tag,
                tags=tags,
                param_count=param_count,
                family=family,
                existing_anchors=existing_anchors,
                config_summary=config_summary,
                card_excerpt=card_excerpt,
                capability_candidates=cap_candidates,
                domain_candidates=dom_candidates,
            )

            shard_idx = idx % num_shards
            shard_files[shard_idx].write(
                json.dumps({
                    "model_id": mid,
                    "prompt": prompt,
                    "valid_anchors": all_valid,
                }) + "\n"
            )
    finally:
        for f in shard_files:
            f.close()

    logger.info(
        "export_c2: wrote %d prompts across %d shards in %s",
        len(model_ids),
        num_shards,
        PHASE_C_WORK_DIR,
    )
    return len(model_ids)


# ---------------------------------------------------------------------------
# Merge C2
# ---------------------------------------------------------------------------


def merge_c2(conn: sqlite3.Connection, files: list[str]) -> dict[str, int]:
    """Merge C2 vibe results into network DB.

    Reads JSONL with {"model_id", "summary", "selected_anchors"} records.
    Stores qwen_summary metadata + links existing dictionary anchors
    (confidence=0.5). Only anchors already in the DB are linked — no new
    anchors are created from C2 output.
    Returns {"merged": N, "skipped": N, "errors": N, "anchors_linked": N}.
    """
    from . import db

    merged = 0
    skipped = 0
    errors = 0
    anchors_linked = 0

    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                model_id = item.get("model_id", "")
                if not model_id:
                    errors += 1
                    continue

                if "error" in item:
                    skipped += 1
                    continue

                summary = item.get("summary", "")
                if not summary:
                    skipped += 1
                    continue

                db.set_metadata(conn, model_id, "qwen_summary", summary, "str")

                # Link selected anchors — only if they exist in the dictionary
                selected = item.get("selected_anchors") or item.get("extra_anchors") or []
                for anchor_label in selected[:5]:
                    anchor_label = anchor_label.strip().lower()
                    if not anchor_label:
                        continue
                    row = conn.execute(
                        "SELECT anchor_id FROM anchors WHERE label = ?",
                        (anchor_label,),
                    ).fetchone()
                    if row:
                        db.link_anchor(
                            conn, model_id, row[0], confidence=0.5
                        )
                        anchors_linked += 1

                merged += 1

    conn.commit()
    logger.info(
        "merge_c2: merged=%d skipped=%d errors=%d anchors_linked=%d",
        merged, skipped, errors, anchors_linked,
    )
    return {"merged": merged, "skipped": skipped, "errors": errors, "anchors_linked": anchors_linked}


# ---------------------------------------------------------------------------
# Export C3
# ---------------------------------------------------------------------------


def export_c3(
    conn: sqlite3.Connection,
    num_shards: int = 4,
) -> int:
    """Export C3 quality gate prompts to sharded JSONL files.

    Queries models that have vibe_summary but no quality_score.
    Builds blind quality gate prompts and writes to
    PHASE_C3_WORK_DIR/shard_N.jsonl.
    """
    # Models with vibe_summary but no quality_score
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

    shard_files = []
    for i in range(num_shards):
        shard_files.append(open(PHASE_C3_WORK_DIR / f"shard_{i}.jsonl", "w"))

    try:
        for idx, row in enumerate(rows):
            model_id = row[0]
            summary = row[1]
            tags = _get_all_anchor_labels(conn, model_id)

            prompt = build_quality_gate_prompt(
                model_id=model_id,
                summary=summary,
                tags=tags,
            )

            shard_idx = idx % num_shards
            shard_files[shard_idx].write(
                json.dumps({"model_id": model_id, "prompt": prompt}) + "\n"
            )
    finally:
        for f in shard_files:
            f.close()

    count = len(rows)
    logger.info(
        "export_c3: wrote %d prompts across %d shards in %s",
        count,
        num_shards,
        PHASE_C3_WORK_DIR,
    )
    return count


# ---------------------------------------------------------------------------
# Merge C3
# ---------------------------------------------------------------------------


def merge_c3(conn: sqlite3.Connection, files: list[str]) -> dict[str, int]:
    """Merge C3 quality gate results into network DB.

    Reads JSONL with {"model_id", "quality_score", "specificity",
    "coherence", "artifacts", "flags"} records.
    Returns {"merged": N, "skipped": N, "errors": N, "passed": N, "failed": N}.
    """
    from . import db

    merged = 0
    skipped = 0
    errors = 0
    passed = 0
    failed = 0

    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                model_id = item.get("model_id", "")
                if not model_id:
                    errors += 1
                    continue

                if "error" in item:
                    skipped += 1
                    continue

                quality_score = item.get("quality_score")
                if quality_score is None:
                    skipped += 1
                    continue

                db.set_metadata(
                    conn, model_id, "quality_score",
                    str(quality_score), "float",
                )

                # Store sub-scores
                for sub_key in ("specificity", "coherence", "artifacts"):
                    val = item.get(sub_key)
                    if val is not None:
                        db.set_metadata(
                            conn, model_id, f"quality_{sub_key}",
                            str(val), "int",
                        )

                # Store flags
                flags = item.get("flags", [])
                if flags:
                    db.set_metadata(
                        conn, model_id, "quality_flags",
                        json.dumps(flags), "json",
                    )

                if quality_score >= QUALITY_GATE_MIN_SCORE:
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


# ---------------------------------------------------------------------------
# Select summaries
# ---------------------------------------------------------------------------


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
        selected, smol_count, qwen_count, skipped,
    )
    return {
        "selected": selected,
        "smol": smol_count,
        "qwen": qwen_count,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


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
    print(f"C3 quality_score:     {status['quality_score']} (pass={status['quality_passing']} fail={status['quality_failing']})")
    print(f"Selected vibe_summary:{status['vibe_summary']}")
