"""Extraction pipeline orchestrator.

Takes raw HF API model data, runs it through all three extraction tiers,
and writes the results into the semantic network database.
"""

from __future__ import annotations

import logging
import sqlite3

from .. import db
from .deterministic import (
    AnchorTag,
    DeterministicResult,
    ModelInput,
)
from .deterministic import (
    extract as extract_deterministic,
)
from .benchmarks import derive_benchmark_anchors, extract_benchmarks
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

    # Extract enrichment data from Tier 1+2 for vibe prompt
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

    # Write bank positions (merge deterministic + pattern results)
    _store_positions(conn, inp.model_id, det, pat)

    # Write anchors (deduplicated from both tiers, with provenance)
    _store_anchors(
        conn,
        inp.model_id,
        det.anchors,
        source="deterministic",
        default_confidence=1.0,
    )
    _store_anchors(
        conn,
        inp.model_id,
        pat.anchors,
        source="pattern",
        default_confidence=0.8,
    )

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
        _store_anchors(
            conn,
            inp.model_id,
            bench_anchors,
            source="benchmark",
            default_confidence=0.75,
        )

        # Card quality score
        from .patterns import _compute_card_quality

        card_quality = _compute_card_quality(card_text)
        if card_quality > 0:
            db.set_metadata(
                conn, inp.model_id, "card_quality", str(card_quality), "float"
            )

    # Inference hardware requirement estimate
    if param_b:
        param_val = float(param_b[0])
        quant = pat.metadata.get("quantization_level", ("", ""))[0]
        if quant and any(q in quant.upper() for q in ("Q4", "Q5", "Q3", "Q2")):
            hw_req = "consumer-GPU" if param_val <= 13 else "high-end-GPU"
        elif param_val <= 3:
            hw_req = "consumer-GPU"
        elif param_val <= 13:
            hw_req = "mid-range-GPU"
        elif param_val <= 70:
            hw_req = "high-end-GPU"
        else:
            hw_req = "multi-GPU"
        db.set_metadata(conn, inp.model_id, "inference_hardware_req", hw_req, "str")

    # Store lineage links for all detected base models
    for base_id, relation in pat.base_models:
        # Ensure target model exists (create stub if not yet indexed)
        existing = conn.execute(
            "SELECT 1 FROM models WHERE model_id = ?", (base_id,)
        ).fetchone()
        if not existing:
            db.insert_model(conn, base_id, source="stub")
        db.add_link(conn, inp.model_id, base_id, relation)


def _store_positions(
    conn: sqlite3.Connection,
    model_id: str,
    det: DeterministicResult,
    pat: PatternResult,
) -> None:
    """Write all 8 bank positions to the database."""
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
    db.set_position(
        conn, model_id, "TRAINING", pat.training.sign, pat.training.depth
    )


def _store_anchors(
    conn: sqlite3.Connection,
    model_id: str,
    anchors: list[AnchorTag],
    source: str = "deterministic",
    default_confidence: float = 1.0,
) -> None:
    """Deduplicate and write anchor links with provenance.

    Uses the per-anchor confidence from AnchorTag when it differs from
    the default (1.0), otherwise falls back to default_confidence.
    """
    seen: set[str] = set()
    for anchor in anchors:
        if anchor.label in seen:
            continue
        seen.add(anchor.label)
        anchor_id = db.get_or_create_anchor(
            conn,
            anchor.label,
            anchor.bank,
            source=source,
        )
        # Use per-anchor confidence if set, else fall back to default
        conf = anchor.confidence if anchor.confidence is not None else default_confidence
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
            for b in siblings[i + 1:]:
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
            for b in model_ids[i + 1:]:
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
            for b in model_ids[i + 1:]:
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
