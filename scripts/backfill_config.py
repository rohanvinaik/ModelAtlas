"""Backfill config.json for all rows in ingest_models that lack it.

Walks ingest_models, downloads config.json for each model that doesn't
already have one in its raw_json, and patches raw_json in-place. Marks
phase_b_done=0 on successful update so Phase B re-extraction picks them
up.

Resumable: skips rows that already have non-null config. Throttled to
stay friendly to the HF API. Logs progress every 100 rows.

Usage:
    python scripts/backfill_config.py                  # full run
    python scripts/backfill_config.py --limit 100      # smoke test
    python scripts/backfill_config.py --throttle 0.5   # adjust pacing
    python scripts/backfill_config.py --dry-run        # report only
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("backfill_config")


def needs_config(raw_json: str) -> bool:
    try:
        raw = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return False
    cfg = raw.get("config")
    return not cfg  # None, missing, or empty dict


def fetch_config_via_adapter(model_id: str, adapter) -> dict | None:
    try:
        return adapter.fetch_config(model_id)
    except Exception as e:  # pragma: no cover — network errors
        logger.debug("fetch_config failed for %s: %s", model_id, e)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ingest-db",
        type=Path,
        default=Path.home() / ".cache" / "model-atlas" / "ingest_state.db",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.3,
        help="Seconds between fetches (default 0.3 = ~3 req/s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N successful backfills (0 = unlimited).",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=50,
        help="Commit SQLite transaction every N rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned work without making any changes.",
    )
    parser.add_argument(
        "--reset-phase-b",
        action="store_true",
        default=True,
        help="Set phase_b_done=0 when config is backfilled (default True).",
    )
    args = parser.parse_args(argv)

    from model_atlas.sources.huggingface import HuggingFaceAdapter

    adapter = HuggingFaceAdapter()

    conn = sqlite3.connect(str(args.ingest_db))
    conn.row_factory = sqlite3.Row

    # Identify candidate rows
    rows = conn.execute(
        """
        SELECT model_id, raw_json
        FROM ingest_models
        WHERE source = 'huggingface'
        ORDER BY likes DESC
        """
    ).fetchall()
    total = len(rows)
    candidates = [(r["model_id"], r["raw_json"]) for r in rows if needs_config(r["raw_json"])]
    logger.info(
        "Backfill plan: %d total rows, %d need config (%d already have it)",
        total,
        len(candidates),
        total - len(candidates),
    )

    if args.dry_run:
        logger.info("Dry-run — no fetches, no writes")
        return 0

    written = 0
    failed = 0
    started = time.time()

    for idx, (model_id, raw_json) in enumerate(candidates, start=1):
        if args.limit and written >= args.limit:
            logger.info("Hit --limit %d, stopping early", args.limit)
            break

        config = fetch_config_via_adapter(model_id, adapter)
        if config is None:
            failed += 1
        else:
            try:
                raw = json.loads(raw_json)
            except (json.JSONDecodeError, TypeError):
                failed += 1
                continue
            raw["config"] = config

            if args.reset_phase_b:
                conn.execute(
                    "UPDATE ingest_models SET raw_json = ?, phase_b_done = 0 WHERE model_id = ?",
                    (json.dumps(raw), model_id),
                )
            else:
                conn.execute(
                    "UPDATE ingest_models SET raw_json = ? WHERE model_id = ?",
                    (json.dumps(raw), model_id),
                )
            written += 1

        if (idx % args.commit_every) == 0:
            conn.commit()
            elapsed = time.time() - started
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta_sec = (len(candidates) - idx) / rate if rate > 0 else 0.0
            logger.info(
                "Progress %d/%d (%.1f%%)  written=%d  failed=%d  rate=%.1f/s  eta=%.1f min",
                idx,
                len(candidates),
                100.0 * idx / len(candidates),
                written,
                failed,
                rate,
                eta_sec / 60.0,
            )

        time.sleep(args.throttle)

    conn.commit()
    elapsed = time.time() - started
    logger.info(
        "Backfill complete: written=%d, failed=%d, elapsed=%.1f min",
        written,
        failed,
        elapsed / 60.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
