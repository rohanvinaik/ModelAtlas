"""Phase C: Vibe extraction via Outlines + local LLM.

Extracts structured summaries and capability anchors from model metadata
using a local language model with constrained generation.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from . import db
from .config import INGEST_VIBE_MIN_LIKES, VIBE_MAX_RETRIES

if TYPE_CHECKING:
    # Type-only: `extraction.vibes` stays a lazy import at the call site so
    # the heavy Outlines/transformers stack is never pulled in on the
    # non-Phase-C paths. `from __future__ import annotations` keeps these
    # names out of the runtime module namespace.
    from .extraction.vibes import VibeExtractor, VibeOutput

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store_vibe_result(
    network_conn: sqlite3.Connection,
    model_id: str,
    result: VibeOutput,
) -> None:
    """Persist vibe extraction results (summary + selected anchors)."""
    if result.summary:
        db.set_metadata(
            network_conn,
            model_id,
            "vibe_summary",
            result.summary,
            "str",
        )
    # `selected_anchors` is the field `VibeExtractor.extract()` actually
    # returns. This read was `result.extra_anchors` — the pre-v0.3 name,
    # which VibeOutput has never carried, so Phase C raised AttributeError
    # the moment it stored a result. The dict-based merge paths
    # (`phase_c_worker`, `ingest_phase_c_merge`) accept either spelling, so
    # the old name is still honoured here for a result object that has it.
    anchors: list[str] = result.selected_anchors or getattr(
        result, "extra_anchors", []
    )
    for anchor_label in anchors:
        anchor_label = anchor_label.strip().lower()
        if anchor_label:
            from .admin import ensure_anchor

            anchor_id = ensure_anchor(
                network_conn,
                anchor_label,
                "CAPABILITY",
                source="vibe",
                reason=f"vibe extraction for {model_id}",
            )
            db.link_anchor(network_conn, model_id, anchor_id, confidence=0.5)


def _extract_single_vibe(
    network_conn: sqlite3.Connection,
    model_id: str,
    raw: dict,
    extractor: VibeExtractor,
    build_vibe_prompt: Callable[..., str],
) -> VibeOutput:
    """Build prompt from pre-extracted data and run vibe extraction."""
    capabilities = _get_model_capabilities(network_conn, model_id)
    family = _get_model_family(network_conn, model_id)
    param_count = _get_param_count(network_conn, model_id)

    prompt = build_vibe_prompt(
        model_id=raw.get("model_id", model_id),
        author=raw.get("author", ""),
        pipeline_tag=raw.get("pipeline_tag", ""),
        tags=raw.get("tags", []),
        param_count=param_count,
        family=family,
        capabilities=capabilities,
    )
    return extractor.extract(prompt)


def phase_c(
    ingest_conn: sqlite3.Connection,
    network_conn: sqlite3.Connection,
    vibe_min_likes: int = INGEST_VIBE_MIN_LIKES,
    *,
    is_shutdown: Callable[[], bool] | None = None,
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
