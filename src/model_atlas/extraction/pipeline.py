"""Extraction pipeline orchestrator.

Takes raw HF API model data, runs it through all three extraction tiers,
and writes the results into the semantic network database.
"""

from __future__ import annotations

import logging
import sqlite3

from .. import db
from .deterministic import (
    DeterministicResult,
    ModelInput,
)
from .deterministic import (
    extract as extract_deterministic,
)
from .patterns import PatternResult
from .patterns import extract as extract_patterns
from .vibes import extract_vibe_summary

logger = logging.getLogger(__name__)


def extract_and_store(
    conn: sqlite3.Connection, inp: ModelInput, card_text: str = ""
) -> None:
    """Run full extraction pipeline and store results in the network.

    This is the main entry point for indexing a single model. It:
    1. Inserts/updates the model entity
    2. Runs tier-1 deterministic extraction (API fields -> positions)
    3. Runs tier-2 pattern extraction (tags/names -> anchors/positions)
    4. Runs tier-3 vibe extraction (card text -> vibe_summary)
    5. Writes all positions, anchors, metadata, and links to the DB
    """
    # Insert model entity
    db.insert_model(conn, inp.model_id, author=inp.author, source=inp.source)

    # Tier 1: Deterministic
    det = extract_deterministic(inp)

    # Tier 2: Patterns
    pat = extract_patterns(
        model_id=inp.model_id,
        author=inp.author,
        tags=inp.tags,
        library_name=inp.library_name,
        pipeline_tag=inp.pipeline_tag,
    )

    # Tier 3: Vibes
    vibe = extract_vibe_summary(
        model_id=inp.model_id,
        card_text=card_text,
        pipeline_tag=inp.pipeline_tag,
        tags=inp.tags,
        author=inp.author,
    )

    # Write bank positions (merge deterministic + pattern results)
    _store_positions(conn, inp.model_id, det, pat)

    # Write anchors (deduplicated from both tiers, with provenance)
    _store_anchors(
        conn,
        inp.model_id,
        det.anchors,
        source="deterministic",
        confidence=1.0,
    )
    _store_anchors(
        conn,
        inp.model_id,
        pat.anchors,
        source="pattern",
        confidence=0.8,
    )

    # Write metadata (deterministic + pattern)
    for key, (value, value_type) in det.metadata.items():
        db.set_metadata(conn, inp.model_id, key, value, value_type)
    for key, (value, value_type) in pat.metadata.items():
        db.set_metadata(conn, inp.model_id, key, value, value_type)

    # Store vibe summary
    if vibe:
        db.set_metadata(conn, inp.model_id, "vibe_summary", vibe, "str")

    # Store lineage link if base model detected
    if pat.base_model:
        db.add_link(conn, inp.model_id, pat.base_model, "fine_tuned_from")


def _store_positions(
    conn: sqlite3.Connection,
    model_id: str,
    det: DeterministicResult,
    pat: PatternResult,
) -> None:
    """Write all 7 bank positions to the database."""
    db.set_position(
        conn,
        model_id,
        "ARCHITECTURE",
        det.architecture.sign,
        det.architecture.depth,
        det.architecture.nodes,
    )
    db.set_position(
        conn, model_id, "EFFICIENCY", det.efficiency.sign, det.efficiency.depth
    )
    db.set_position(conn, model_id, "QUALITY", det.quality.sign, det.quality.depth)
    db.set_position(
        conn, model_id, "CAPABILITY", pat.capability.sign, pat.capability.depth
    )
    db.set_position(
        conn, model_id, "COMPATIBILITY", pat.compatibility.sign, pat.compatibility.depth
    )
    db.set_position(conn, model_id, "LINEAGE", pat.lineage.sign, pat.lineage.depth)
    db.set_position(conn, model_id, "DOMAIN", pat.domain.sign, pat.domain.depth)


def _store_anchors(
    conn: sqlite3.Connection,
    model_id: str,
    anchors: list[tuple[str, str]],
    source: str = "deterministic",
    confidence: float = 1.0,
) -> None:
    """Deduplicate and write anchor links with provenance."""
    seen: set[str] = set()
    for label, bank_name in anchors:
        if label in seen:
            continue
        seen.add(label)
        anchor_id = db.get_or_create_anchor(
            conn,
            label,
            bank_name,
            source=source,
        )
        db.link_anchor(conn, model_id, anchor_id, confidence=confidence)


def extract_batch(conn: sqlite3.Connection, models: list[dict]) -> int:
    """Extract and store a batch of models. Returns count of models processed."""
    count = 0
    for m in models:
        try:
            inp = ModelInput(
                model_id=m.get("model_id", ""),
                author=m.get("author", ""),
                pipeline_tag=m.get("pipeline_tag", ""),
                tags=m.get("tags", []),
                library_name=m.get("library_name", ""),
                likes=m.get("likes", 0),
                downloads=m.get("downloads", 0),
                created_at=m.get("created_at"),
                license_str=m.get("license", ""),
                safetensors_info=m.get("safetensors_info"),
                config=m.get("config"),
            )
            extract_and_store(conn, inp, card_text=m.get("card_text", ""))
            count += 1
        except Exception:
            logger.warning(
                "Failed to extract model: %s",
                m.get("model_id", "?"),
                exc_info=True,
            )
    return count
