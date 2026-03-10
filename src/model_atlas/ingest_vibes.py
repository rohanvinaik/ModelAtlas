"""Phase C: Vibe extraction via Outlines + local LLM.

Extracts structured summaries and capability anchors from model metadata
using a local language model with constrained generation.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

from . import db
from .config import INGEST_VIBE_MIN_LIKES, VIBE_MAX_RETRIES

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store_vibe_result(
    network_conn: sqlite3.Connection,
    model_id: str,
    result: object,
) -> None:
    """Persist vibe extraction results (summary + extra anchors)."""
    if result.summary:  # type: ignore[union-attr]
        db.set_metadata(
            network_conn,
            model_id,
            "vibe_summary",
            result.summary,
            "str",  # type: ignore[union-attr]
        )
    for anchor_label in result.extra_anchors:  # type: ignore[union-attr]
        anchor_label = anchor_label.strip().lower()
        if anchor_label:
            anchor_id = db.get_or_create_anchor(
                network_conn, anchor_label, "CAPABILITY", source="vibe"
            )
            db.link_anchor(network_conn, model_id, anchor_id, confidence=0.5)


def _extract_single_vibe(
    network_conn: sqlite3.Connection,
    model_id: str,
    raw: dict,
    extractor: object,
    build_vibe_prompt: object,
) -> object:
    """Build prompt from pre-extracted data and run vibe extraction."""
    capabilities = _get_model_capabilities(network_conn, model_id)
    family = _get_model_family(network_conn, model_id)
    param_count = _get_param_count(network_conn, model_id)

    prompt = build_vibe_prompt(  # type: ignore[operator]
        model_id=raw.get("model_id", model_id),
        author=raw.get("author", ""),
        pipeline_tag=raw.get("pipeline_tag", ""),
        tags=raw.get("tags", []),
        param_count=param_count,
        family=family,
        capabilities=capabilities,
    )
    return extractor.extract(prompt)  # type: ignore[union-attr]


def phase_c(
    ingest_conn: sqlite3.Connection,
    network_conn: sqlite3.Connection,
    vibe_min_likes: int = INGEST_VIBE_MIN_LIKES,
    *,
    is_shutdown: object = None,
) -> int:
    """Run Outlines-based vibe extraction on eligible models.

    Args:
        is_shutdown: Optional callable returning bool, checked for graceful shutdown.
    """
    from .extraction.vibes import VibeExtractor, build_vibe_prompt

    cursor = ingest_conn.execute(
        """SELECT model_id, raw_json FROM ingest_models
           WHERE phase_b_done = 1 AND phase_c_done = 0
             AND phase_c_attempts < ? AND likes >= ?
           ORDER BY likes DESC""",
        (VIBE_MAX_RETRIES, vibe_min_likes),
    )

    extractor = VibeExtractor()
    extractor.load()

    count = 0
    for row in cursor:
        if is_shutdown and is_shutdown():
            break

        model_id = row["model_id"]
        try:
            raw = json.loads(row["raw_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        ingest_conn.execute(
            "UPDATE ingest_models SET phase_c_attempts = phase_c_attempts + 1 WHERE model_id = ?",
            (model_id,),
        )

        try:
            result = _extract_single_vibe(
                network_conn,
                model_id,
                raw,
                extractor,
                build_vibe_prompt,
            )
            _store_vibe_result(network_conn, model_id, result)

            ingest_conn.execute(
                """UPDATE ingest_models
                   SET phase_c_done = 1, vibed_at = ?
                   WHERE model_id = ?""",
                (_now_iso(), model_id),
            )
            count += 1

            if count % 10 == 0:
                network_conn.commit()
                ingest_conn.commit()
                logger.info("Phase C: %d models vibed...", count)

        except Exception:
            logger.warning("Phase C: failed %s", model_id, exc_info=True)

    network_conn.commit()
    ingest_conn.commit()
    logger.info("Phase C: complete — %d models vibed", count)
    return count


def _get_model_capabilities(conn: sqlite3.Connection, model_id: str) -> list[str]:
    """Get capability anchors for a model from the network DB."""
    rows = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ? AND a.bank = 'CAPABILITY'""",
        (model_id,),
    ).fetchall()
    return [r[0] for r in rows]


def _get_model_family(conn: sqlite3.Connection, model_id: str) -> str:
    """Get family anchor for a model."""
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
        """SELECT value FROM model_metadata
           WHERE model_id = ? AND key = 'parameter_count_b'""",
        (model_id,),
    ).fetchone()
    if row:
        return f"{row[0]}B parameters"
    return "unknown"
