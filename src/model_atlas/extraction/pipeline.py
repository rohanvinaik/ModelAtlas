"""Extraction pipeline orchestrator.

Takes raw HF API model data, runs it through all three extraction tiers,
and writes the results into the semantic network database.

Canonical writes (``models``, ``model_positions``, ``model_links``,
``anchors``) are dispatched through
:func:`model_atlas.reconciler.reconcile_items` so every change is
audit-logged with the extraction context as its reason. Observation
writes (``model_metadata``, ``model_anchors`` assignments) remain
direct — those tables are Mode 2 in the bi-modal split.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .. import db
from ..admin import ensure_anchor
from ..reconciler import reconcile_items
from .benchmarks import derive_benchmark_anchors, extract_benchmarks
from .deterministic import (
    AnchorTag,
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


def _utc_iso_z() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def extract_and_store(
    conn: sqlite3.Connection, inp: ModelInput, card_text: str = ""
) -> None:
    """Run full extraction pipeline and store results in the network.

    This is the main entry point for indexing a single model. It:

    1. Inserts/updates the model entity (canonical, audit-logged)
    2. Runs tier-1 deterministic extraction (API fields -> positions)
    3. Runs tier-2 pattern extraction (tags/names -> anchors/positions)
    4. Runs tier-3 vibe extraction (card text -> vibe_summary)
    5. Writes positions and lineage links via the reconciler (canonical)
    6. Writes metadata and anchor assignments directly (observation)
    """
    captured_at = _utc_iso_z()

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

    # Extract enrichment data from Tier 1+2 for vibe prompt
    param_count, family, capabilities, training_methods = _extract_enrichment_context(det, pat)
    param_b = det.metadata.get("parameter_count_b")

    # Tier 3: Vibes
    vibe = extract_vibe_summary(
        model_id=inp.model_id,
        card_text=card_text,
        pipeline_tag=inp.pipeline_tag,
        tags=inp.tags,
        author=inp.author,
        param_count=param_count,
        family=family,
        capabilities=capabilities,
        training_method=", ".join(training_methods) if training_methods else "unknown",
    )

    # --- Build canonical items for reconciler dispatch ---
    items = _build_canonical_items(inp, det, pat, captured_at)
    reconcile_items(items, conn, apply=True, source_label="extract_and_store")

    # --- Mode 2: observation writes (direct, not audit-logged) ---

    # Write anchors (deduplicated from both tiers, with provenance)
    _link_anchors(conn, inp.model_id, det.anchors, source="deterministic",
                  default_confidence=1.0)
    _link_anchors(conn, inp.model_id, pat.anchors, source="pattern",
                  default_confidence=0.8)

    # Write metadata (deterministic + pattern)
    for key, (value, value_type) in det.metadata.items():
        db.set_metadata(conn, inp.model_id, key, value, value_type)
    for key, (value, value_type) in pat.metadata.items():
        db.set_metadata(conn, inp.model_id, key, value, value_type)

    # Store vibe summary
    if vibe:
        db.set_metadata(conn, inp.model_id, "vibe_summary", vibe, "str")

    # Extract and store benchmark scores from card text
    if card_text:
        benchmarks = extract_benchmarks(card_text)
        for key, (value, value_type) in benchmarks.items():
            db.set_metadata(conn, inp.model_id, key, value, value_type)

        # Derive QUALITY anchors from benchmark thresholds
        bench_anchors = derive_benchmark_anchors(benchmarks)
        _link_anchors(conn, inp.model_id, bench_anchors, source="benchmark",
                      default_confidence=0.75)

        # Card quality score
        from .patterns import _compute_card_quality

        card_quality = _compute_card_quality(card_text)
        if card_quality > 0:
            db.set_metadata(
                conn, inp.model_id, "card_quality", str(card_quality), "float"
            )

    # Inference hardware requirement estimate
    if param_b:
        quant = pat.metadata.get("quantization_level", ("", ""))[0]
        hw_req = _infer_hardware_requirement(param_b[0], quant)
        db.set_metadata(conn, inp.model_id, "inference_hardware_req", hw_req, "str")


def _build_canonical_items(
    inp: ModelInput,
    det: DeterministicResult,
    pat: PatternResult,
    captured_at: str,
) -> list[dict[str, Any]]:
    """Build the list of reconciler-shaped items for one extraction."""
    items: list[dict[str, Any]] = []

    # Primary model entity
    display_name = inp.model_id.split("/")[-1]
    items.append(
        {
            "op": "upsert",
            "table": "models",
            "key": {"model_id": inp.model_id},
            "row": {
                "author": inp.author,
                "source": inp.source,
                "display_name": display_name,
            },
            "host": "extract_and_store",
            "captured_at": captured_at,
        }
    )

    # 8 bank positions
    position_specs: list[tuple[str, int, int, list[str] | None]] = [
        ("ARCHITECTURE", det.architecture.sign, det.architecture.depth, det.architecture.nodes),
        ("EFFICIENCY", det.efficiency.sign, det.efficiency.depth, None),
        ("QUALITY", det.quality.sign, det.quality.depth, None),
        ("CAPABILITY", pat.capability.sign, pat.capability.depth, None),
        ("COMPATIBILITY", pat.compatibility.sign, pat.compatibility.depth, None),
        ("LINEAGE", pat.lineage.sign, pat.lineage.depth, None),
        ("DOMAIN", pat.domain.sign, pat.domain.depth, None),
        ("TRAINING", pat.training.sign, pat.training.depth, None),
    ]
    from ..db import ZERO_STATES

    for bank, sign, depth, nodes in position_specs:
        row: dict[str, Any] = {
            "path_sign": sign,
            "path_depth": depth,
            "path_nodes": json.dumps(nodes) if nodes else None,
            "zero_state": ZERO_STATES.get(bank, ""),
        }
        items.append(
            {
                "op": "upsert",
                "table": "model_positions",
                "key": {"model_id": inp.model_id, "bank": bank},
                "row": row,
                "host": "extract_and_store",
                "captured_at": captured_at,
            }
        )

    # Lineage links: stub the target model if absent, then add the link
    for base_id, relation in pat.base_models:
        items.append(
            {
                "op": "upsert",
                "table": "models",
                "key": {"model_id": base_id},
                "row": {"source": "stub", "author": "", "display_name": base_id.split("/")[-1]},
                "host": "extract_and_store",
                "captured_at": captured_at,
            }
        )
        items.append(
            {
                "op": "upsert",
                "table": "model_links",
                "key": {
                    "source_id": inp.model_id,
                    "target_id": base_id,
                    "relation": relation,
                },
                "row": {"weight": 1.0},
                "host": "extract_and_store",
                "captured_at": captured_at,
            }
        )

    return items


def _link_anchors(
    conn: sqlite3.Connection,
    model_id: str,
    anchors: list[AnchorTag],
    source: str,
    default_confidence: float,
) -> None:
    """Ensure anchor vocabulary exists (canonical, audit-logged) and link
    the model to each (observation, direct write).
    """
    seen: set[str] = set()
    for anchor in anchors:
        if anchor.label in seen:
            continue
        seen.add(anchor.label)
        anchor_id = ensure_anchor(
            conn,
            anchor.label,
            anchor.bank,
            source=source,
            reason=f"{source} extraction tier for {model_id}",
        )
        conf = (
            anchor.confidence if anchor.confidence is not None else default_confidence
        )
        db.link_anchor(conn, model_id, anchor_id, confidence=conf)


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
    if count > 1:
        infer_relationships(conn)
    return count


# --- Post-batch relationship inference ---


def _infer_sibling_links(conn: sqlite3.Connection) -> int:
    """Create variant_of links between models that share the same base model.

    Uses SQL GROUP BY on model_links target_id to find siblings.
    Caps at 20 siblings per group to avoid combinatorial explosion.
    """
    rows = conn.execute(
        """SELECT target_id, GROUP_CONCAT(source_id) AS sources
           FROM model_links
           WHERE relation IN ('fine_tuned_from', 'quantized_from', 'merged_from')
           GROUP BY target_id
           HAVING COUNT(source_id) > 1"""
    ).fetchall()

    count = 0
    for row in rows:
        siblings = row[1].split(",")[:20]  # cap
        for i, a in enumerate(siblings):
            for b in siblings[i + 1 :]:
                conn.execute(
                    """INSERT OR IGNORE INTO model_links (source_id, target_id, relation, weight)
                       VALUES (?, ?, 'variant_of', 0.6)""",
                    (a, b),
                )
                count += 1
    return count


def _infer_variant_links(conn: sqlite3.Connection) -> int:
    """Create variant_of links between models from the same author with shared name prefix.

    Groups by author, then checks for shared name prefix >= 3 chars.
    Caps at 50 models per author to avoid O(n^2) blowup.
    """
    rows = conn.execute(
        """SELECT author, GROUP_CONCAT(model_id) AS models
           FROM models
           WHERE author != ''
           GROUP BY author
           HAVING COUNT(model_id) > 1"""
    ).fetchall()

    count = 0
    for row in rows:
        model_ids = row[1].split(",")[:50]  # cap
        for i, a in enumerate(model_ids):
            name_a = a.split("/")[-1].lower()
            for b in model_ids[i + 1 :]:
                name_b = b.split("/")[-1].lower()
                # Find shared prefix length
                prefix_len = 0
                for ca, cb in zip(name_a, name_b):
                    if ca != cb:
                        break
                    prefix_len += 1
                if prefix_len >= 3:
                    conn.execute(
                        """INSERT OR IGNORE INTO model_links
                           (source_id, target_id, relation, weight)
                           VALUES (?, ?, 'variant_of', 0.5)""",
                        (a, b),
                    )
                    count += 1
    return count


def _infer_fingerprint_links(conn: sqlite3.Connection) -> int:
    """Create same_family links between models sharing a structural fingerprint.

    Uses SQL GROUP BY on model_metadata structural_fingerprint.
    """
    rows = conn.execute(
        """SELECT value, GROUP_CONCAT(model_id) AS models
           FROM model_metadata
           WHERE key = 'structural_fingerprint'
           GROUP BY value
           HAVING COUNT(model_id) > 1"""
    ).fetchall()

    count = 0
    for row in rows:
        model_ids = row[1].split(",")[:50]  # cap
        for i, a in enumerate(model_ids):
            for b in model_ids[i + 1 :]:
                conn.execute(
                    """INSERT OR IGNORE INTO model_links
                       (source_id, target_id, relation, weight)
                       VALUES (?, ?, 'same_family', 0.5)""",
                    (a, b),
                )
                count += 1
    return count


def infer_relationships(conn: sqlite3.Connection) -> int:
    """Run all post-batch relationship inference. Returns total links created."""
    total = 0
    total += _infer_sibling_links(conn)
    total += _infer_variant_links(conn)
    total += _infer_fingerprint_links(conn)
    if total:
        logger.info("Inferred %d relationship links", total)
    return total


def _extract_enrichment_context(
    det: DeterministicResult, pat: PatternResult
) -> tuple[str, str, list[str], list[str]]:
    """Extract enrichment data from Tier 1+2 results for vibe prompt.

    Returns (param_count, family, capabilities, training_methods).
    """
    param_b = det.metadata.get("parameter_count_b")
    param_count = param_b[0] + "B" if param_b else "unknown"
    family = "unknown"
    capabilities: list[str] = []
    training_methods: list[str] = []
    for anchor in pat.anchors:
        if anchor.bank == "LINEAGE" and anchor.label.endswith("-family"):
            family = anchor.label
        elif anchor.bank == "CAPABILITY":
            capabilities.append(anchor.label)
        elif anchor.bank == "TRAINING":
            training_methods.append(anchor.label)
    return param_count, family, capabilities, training_methods


def _infer_hardware_requirement(param_b_str: str, quantization: str) -> str:
    """Estimate inference hardware requirement from param count and quantization."""
    param_val = float(param_b_str)
    if quantization and any(q in quantization.upper() for q in ("Q4", "Q5", "Q3", "Q2")):
        return "consumer-GPU" if param_val <= 13 else "high-end-GPU"
    if param_val <= 3:
        return "consumer-GPU"
    if param_val <= 13:
        return "mid-range-GPU"
    if param_val <= 70:
        return "high-end-GPU"
    return "multi-GPU"
