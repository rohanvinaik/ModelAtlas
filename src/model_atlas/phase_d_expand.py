"""D2: Dictionary expansion with strict DSL.

Reads YAML expansion specs and creates new anchor labels in the dictionary,
optionally auto-linking matching models. Three modes:

  - create_only: insert anchor, don't link any models
  - auto_link: create anchor + bulk-link matching models at specified confidence
  - queue_for_heal: create anchor + flag matching models for D3 review

Matcher types (strict, boundary-aware):
  - tag_exact: tag == value (not substring)
  - tag_regex: re.search with word boundaries
  - pipeline_tag_in: pipeline_tag in list
  - name_regex: regex against model_id
  - metadata_equals: model_metadata key=value
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import db

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """Summary of a D2 expansion run."""

    run_id: str
    anchors_created: int = 0
    models_linked: int = 0
    models_queued: int = 0
    per_label: dict[str, dict[str, int]] = field(default_factory=dict)


def _load_spec(spec_path: str | Path) -> list[dict]:
    """Load expansion spec from YAML file."""
    import yaml

    with open(spec_path) as f:
        data = yaml.safe_load(f)
    return data.get("expansions", [])


def _check_condition(
    condition: dict,
    model_id: str,
    tags: set[str],
    pipeline_tag: str,
    metadata: dict[str, str],
) -> bool:
    """Check a single match condition against a model."""
    ctype = condition["type"]
    value = condition["value"]

    if ctype == "tag_exact":
        return value in tags

    if ctype == "tag_regex":
        compiled = re.compile(value, re.IGNORECASE)
        return any(compiled.search(t) for t in tags)

    if ctype == "pipeline_tag_in":
        return pipeline_tag in value

    if ctype == "name_regex":
        return bool(re.search(value, model_id, re.IGNORECASE))

    if ctype == "metadata_equals":
        key, expected = value.split("=", 1)
        return metadata.get(key) == expected

    logger.warning("Unknown condition type: %s", ctype)
    return False


def _matches_model(
    entry: dict,
    model_id: str,
    tags: set[str],
    pipeline_tag: str,
    metadata: dict[str, str],
) -> bool:
    """Check if a model matches the expansion entry's match_rules."""
    rules = entry.get("match_rules", {})
    conditions = rules.get("conditions", [])
    if not conditions:
        return False

    operator = rules.get("operator", "AND")
    min_matches = rules.get("min_matches", 1)

    matches = sum(
        1
        for c in conditions
        if _check_condition(c, model_id, tags, pipeline_tag, metadata)
    )

    if operator == "AND":
        return matches == len(conditions)
    # OR mode
    return matches >= min_matches


def _get_model_tags(conn: sqlite3.Connection, model_id: str) -> set[str]:
    """Get all tags for a model from anchors + raw tags in metadata."""
    # Anchor labels
    rows = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ?""",
        (model_id,),
    ).fetchall()
    tags = {r[0] for r in rows}

    # Raw tags from metadata (stored as JSON list)
    row = conn.execute(
        "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'tags'",
        (model_id,),
    ).fetchone()
    if row and row[0]:
        try:
            raw_tags = json.loads(row[0])
            if isinstance(raw_tags, list):
                tags.update(raw_tags)
        except (json.JSONDecodeError, TypeError):
            pass

    return tags


def _get_model_metadata_dict(
    conn: sqlite3.Connection, model_id: str
) -> dict[str, str]:
    """Get all metadata as a flat dict."""
    rows = conn.execute(
        "SELECT key, value FROM model_metadata WHERE model_id = ?",
        (model_id,),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def expand_dictionary(
    conn: sqlite3.Connection,
    spec_path: str | Path,
    dry_run: bool = False,
) -> ExpansionResult:
    """Run D2 dictionary expansion from a YAML spec.

    Args:
        conn: Network database connection.
        spec_path: Path to expansion YAML spec.
        dry_run: If True, preview counts without writing.

    Returns:
        ExpansionResult with run summary.
    """
    entries = _load_spec(spec_path)
    config = {"spec_path": str(spec_path), "dry_run": dry_run, "entries": len(entries)}

    run_id = ""
    if not dry_run:
        run_id = db.create_phase_d_run(conn, "d2", config=config)

    # Pre-load all models with pipeline_tags
    all_models = conn.execute("SELECT model_id FROM models").fetchall()
    model_ids = [r[0] for r in all_models]

    pipeline_tags: dict[str, str] = {}
    for row in conn.execute(
        "SELECT model_id, value FROM model_metadata WHERE key = 'pipeline_tag'"
    ).fetchall():
        pipeline_tags[row[0]] = row[1]

    result = ExpansionResult(run_id=run_id)

    for entry in entries:
        label = entry["label"]
        bank = entry["bank"]
        category = entry.get("category", "")
        mode = entry.get("mode", "create_only")
        confidence = entry.get("confidence", 0.7)

        # Check if anchor already exists
        existing = conn.execute(
            "SELECT anchor_id FROM anchors WHERE label = ?", (label,)
        ).fetchone()

        label_stats: dict[str, int] = {"matched": 0, "linked": 0, "queued": 0}

        if not existing and not dry_run:
            db.get_or_create_anchor(conn, label, bank, category=category, source="expansion")
            result.anchors_created += 1

        if mode == "create_only":
            result.per_label[label] = label_stats
            continue

        # Find matching models
        for model_id in model_ids:
            tags = _get_model_tags(conn, model_id)
            ptag = pipeline_tags.get(model_id, "")
            metadata = _get_model_metadata_dict(conn, model_id)

            if not _matches_model(entry, model_id, tags, ptag, metadata):
                continue

            label_stats["matched"] += 1

            if dry_run:
                continue

            anchor_id = db.get_or_create_anchor(
                conn, label, bank, category=category, source="expansion"
            )

            if mode == "auto_link":
                db.link_anchor(conn, model_id, anchor_id, confidence=confidence)
                label_stats["linked"] += 1
                result.models_linked += 1

            elif mode == "queue_for_heal":
                db.insert_audit_finding(
                    conn,
                    run_id=run_id,
                    model_id=model_id,
                    mismatch_type="expansion_candidate",
                    bank=bank,
                    det_anchor=label,
                    severity=0.3,
                    detail={"source": "d2_expansion", "label": label},
                )
                label_stats["queued"] += 1
                result.models_queued += 1

        result.per_label[label] = label_stats

        if not dry_run and label_stats.get("linked", 0) > 0:
            conn.commit()

    if not dry_run:
        summary = {
            "anchors_created": result.anchors_created,
            "models_linked": result.models_linked,
            "models_queued": result.models_queued,
            "per_label": result.per_label,
        }
        db.finish_phase_d_run(conn, run_id, "completed", summary)
        conn.commit()

    logger.info(
        "D2 expansion%s: %d anchors created, %d models linked, %d queued",
        " (dry run)" if dry_run else "",
        result.anchors_created,
        result.models_linked,
        result.models_queued,
    )

    return result
