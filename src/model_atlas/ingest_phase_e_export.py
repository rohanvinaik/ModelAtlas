"""Phase E export: build prioritized JSONL input for web enrichment workers.

Queries network.db for models, generates search queries from existing
metadata, and includes per-bank anchor vocabularies for typed skeleton
extraction.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import ExitStack
from pathlib import Path

from .config import PHASE_E_WORK_DIR

logger = logging.getLogger(__name__)

ALL_BANKS = [
    "ARCHITECTURE", "CAPABILITY", "COMPATIBILITY", "DOMAIN",
    "EFFICIENCY", "LINEAGE", "QUALITY", "TRAINING",
]


def _get_anchor_labels_by_bank(conn: sqlite3.Connection, bank: str) -> list[str]:
    rows = conn.execute(
        "SELECT label FROM anchors WHERE bank = ? ORDER BY label",
        (bank,),
    ).fetchall()
    return [r[0] for r in rows]


def _get_all_anchor_labels(conn: sqlite3.Connection, model_id: str) -> list[str]:
    rows = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ?""",
        (model_id,),
    ).fetchall()
    return [r[0] for r in rows]


def _get_metadata(conn: sqlite3.Connection, model_id: str, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM model_metadata WHERE model_id = ? AND key = ?",
        (model_id, key),
    ).fetchone()
    return row[0] if row else None


def _get_author(conn: sqlite3.Connection, model_id: str) -> str:
    row = conn.execute(
        "SELECT author FROM models WHERE model_id = ?", (model_id,)
    ).fetchone()
    return row[0] if row else ""


def _get_family(conn: sqlite3.Connection, model_id: str) -> str:
    row = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ? AND a.bank = 'LINEAGE'
             AND a.category = 'family'
           LIMIT 1""",
        (model_id,),
    ).fetchone()
    return row[0] if row else ""


def _build_search_queries(
    model_id: str,
    author: str,
    pipeline_tag: str,
    family: str,
    param_count: str,
    domain_anchors: list[str],
) -> list[str]:
    """Generate 2-3 search queries for a model."""
    queries = []
    display_name = model_id.split("/")[-1] if "/" in model_id else model_id

    # Query 1: Direct model search for benchmarks/reviews
    q1_parts = [f'"{display_name}"']
    if pipeline_tag:
        q1_parts.append(pipeline_tag)
    q1_parts.append("benchmark review")
    queries.append(" ".join(q1_parts))

    # Query 2: Author + model for evaluation results
    q2_parts = []
    if author:
        q2_parts.append(author)
    q2_parts.append(display_name)
    q2_parts.append("evaluation performance comparison")
    queries.append(" ".join(q2_parts))

    # Query 3 (conditional): Domain/capability-specific
    if domain_anchors:
        domain_str = domain_anchors[0].replace("-domain", "").replace("-code", " code")
        q3 = f"{display_name} {domain_str} model"
        if param_count and param_count != "unknown":
            q3 += f" {param_count}"
        queries.append(q3)
    elif family:
        family_short = family.replace("-family", "")
        queries.append(f"{family_short} {display_name} comparison")

    return queries[:3]


def _build_one_record(
    conn: sqlite3.Connection,
    model_id: str,
    bank_vocab: dict[str, list[str]],
) -> dict:
    """Build one JSONL record for a model."""
    author = _get_author(conn, model_id)
    pipeline_tag = _get_metadata(conn, model_id, "pipeline_tag") or ""
    param_count = _get_metadata(conn, model_id, "parameter_count_b") or ""
    if param_count:
        param_count = f"{param_count}B"
    family = _get_family(conn, model_id)
    vibe_summary = _get_metadata(conn, model_id, "vibe_summary") or ""
    current_anchors = _get_all_anchor_labels(conn, model_id)

    domain_anchors = [
        a for a in current_anchors
        if any(a.endswith(s) for s in ("-domain", "-code"))
    ]

    search_queries = _build_search_queries(
        model_id, author, pipeline_tag, family, param_count, domain_anchors,
    )

    return {
        "model_id": model_id,
        "search_queries": search_queries,
        "existing_metadata": {
            "author": author,
            "pipeline_tag": pipeline_tag,
            "param_count": param_count,
            "family": family,
            "vibe_summary": vibe_summary[:300],
            "current_anchors": current_anchors,
        },
        "banks": bank_vocab,
    }


def get_priority_models(
    conn: sqlite3.Connection,
    min_downloads: int = 100,
    full_corpus: bool = False,
    skip_existing: bool = True,
) -> list[str]:
    """Get models ordered by priority: high-download first, then rest."""
    already_done: set[str] = set()
    if skip_existing:
        rows = conn.execute(
            "SELECT model_id FROM model_metadata WHERE key = 'web_enriched'"
        ).fetchall()
        already_done = {r[0] for r in rows}

    priority_rows = conn.execute(
        """SELECT m.model_id, CAST(COALESCE(dl.value, '0') AS INTEGER) as downloads
           FROM models m
           LEFT JOIN model_metadata dl ON m.model_id = dl.model_id AND dl.key = 'downloads'
           WHERE CAST(COALESCE(dl.value, '0') AS INTEGER) >= ?
           ORDER BY downloads DESC""",
        (min_downloads,),
    ).fetchall()
    priority_ids = [r[0] for r in priority_rows if r[0] not in already_done]

    if not full_corpus:
        return priority_ids

    priority_set = set(priority_ids) | already_done
    rest_rows = conn.execute(
        """SELECT m.model_id, CAST(COALESCE(lk.value, '0') AS INTEGER) as likes
           FROM models m
           LEFT JOIN model_metadata lk ON m.model_id = lk.model_id AND lk.key = 'likes'
           ORDER BY likes DESC"""
    ).fetchall()
    rest_ids = [r[0] for r in rest_rows if r[0] not in priority_set]

    return priority_ids + rest_ids


def export_phase_e(
    conn: sqlite3.Connection,
    num_shards: int = 4,
    banks: list[str] | None = None,
    min_downloads: int = 100,
    full_corpus: bool = False,
    work_dir: Path | None = None,
) -> int:
    """Export Phase E JSONL shards for web enrichment workers.

    Returns number of models exported.
    """
    out_dir = work_dir or PHASE_E_WORK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    target_banks = banks or ALL_BANKS
    bank_vocab: dict[str, list[str]] = {}
    for bank in target_banks:
        labels = _get_anchor_labels_by_bank(conn, bank)
        if labels:
            bank_vocab[bank] = labels

    model_ids = get_priority_models(conn, min_downloads, full_corpus)
    if not model_ids:
        logger.info("export_phase_e: no models to export")
        return 0

    with ExitStack() as stack:
        shard_files = [
            stack.enter_context(open(out_dir / f"shard_{i}.jsonl", "w"))
            for i in range(num_shards)
        ]
        for idx, mid in enumerate(model_ids):
            record = _build_one_record(conn, mid, bank_vocab)
            shard_files[idx % num_shards].write(json.dumps(record) + "\n")

    logger.info(
        "export_phase_e: wrote %d models across %d shards to %s",
        len(model_ids), num_shards, out_dir,
    )
    return len(model_ids)
