"""Phase E export: build prioritized JSONL input for web enrichment workers.

Queries network.db for models, generates search queries from existing
metadata, and includes per-bank anchor vocabularies for typed skeleton
extraction.

Usage:
    python -m model_atlas.ingest --export-e 4
    python -m model_atlas.ingest --export-e 4 --export-e-banks CAPABILITY,DOMAIN,QUALITY
    python -m model_atlas.ingest --export-e 4 --export-e-min-downloads 1000

Or standalone:
    python scripts/export_phase_e.py --num-shards 4 --min-downloads 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from contextlib import ExitStack
from pathlib import Path

logger = logging.getLogger(__name__)

# Default work dir (overridden when called via ingest_cli which uses config)
DEFAULT_WORK_DIR = Path.home() / ".cache" / "model-atlas" / "phase_e_work"

ALL_BANKS = [
    "ARCHITECTURE", "CAPABILITY", "COMPATIBILITY", "DOMAIN",
    "EFFICIENCY", "LINEAGE", "QUALITY", "TRAINING",
]


def _get_anchor_labels_by_bank(conn: sqlite3.Connection, bank: str) -> list[str]:
    """All anchor labels in a bank (dictionary-wide)."""
    rows = conn.execute(
        "SELECT label FROM anchors WHERE bank = ? ORDER BY label",
        (bank,),
    ).fetchall()
    return [r[0] for r in rows]


def _get_all_anchor_labels(conn: sqlite3.Connection, model_id: str) -> list[str]:
    """All anchor labels currently assigned to a model."""
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
    vibe_summary: str,
    domain_anchors: list[str],
) -> list[str]:
    """Generate 2-3 search queries for a model."""
    queries = []

    # Query 1: Direct model search for benchmarks/reviews
    display_name = model_id.split("/")[-1] if "/" in model_id else model_id
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

    # Query 3 (conditional): Domain/capability-specific if we have signals
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

    # Get domain anchors for query building
    domain_anchors = [
        a for a in current_anchors
        if any(a.endswith(s) for s in ("-domain", "-code"))
    ]

    search_queries = _build_search_queries(
        model_id, author, pipeline_tag, family, param_count,
        vibe_summary, domain_anchors,
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
    # Models already web-enriched
    already_done: set[str] = set()
    if skip_existing:
        rows = conn.execute(
            "SELECT model_id FROM model_metadata WHERE key = 'web_enriched'"
        ).fetchall()
        already_done = {r[0] for r in rows}

    # Priority tier 1: models with downloads >= threshold
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

    # Priority tier 2: remaining models
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
    out_dir = work_dir or DEFAULT_WORK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    target_banks = banks or ALL_BANKS
    # Build bank vocabulary once
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


def main() -> None:
    """Standalone CLI for export (also callable via ingest_cli)."""
    parser = argparse.ArgumentParser(description="Export Phase E web enrichment input")
    parser.add_argument("--db", default=str(Path.home() / ".cache/model-atlas/network.db"))
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--banks", default=None, help="Comma-separated bank list")
    parser.add_argument("--min-downloads", type=int, default=100)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    bank_list = args.banks.upper().split(",") if args.banks else None
    work_dir = Path(args.output_dir) if args.output_dir else None

    n = export_phase_e(
        conn,
        num_shards=args.num_shards,
        banks=bank_list,
        min_downloads=args.min_downloads,
        full_corpus=args.full_corpus,
        work_dir=work_dir,
    )
    conn.close()
    print(f"Exported {n} models across {args.num_shards} shards")


if __name__ == "__main__":
    main()
