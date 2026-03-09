"""Backfill ingest_state.db with raw_json from HuggingFace API.

Reads model_ids from network.db, fetches model info via huggingface_hub,
and stores serialized raw_json in ingest_state.db.

Resumable: skips models that already have raw_json.
Rate-limit aware: sequential with backoff on 429s.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

from huggingface_hub import HfApi, ModelInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "model-atlas"
NETWORK_DB = CACHE_DIR / "network.db"
INGEST_DB = CACHE_DIR / "ingest_state.db"

BATCH_COMMIT = 100
MAX_RETRIES = 5
BASE_BACKOFF = 10.0  # seconds — HF rate limits are aggressive


def model_info_to_raw(info: ModelInfo) -> dict:
    """Convert ModelInfo to a raw_json dict matching HF API format."""
    return {
        "model_id": info.id or "",
        "author": info.author or "",
        "pipeline_tag": info.pipeline_tag or "",
        "tags": list(info.tags or []),
        "library_name": info.library_name or "",
        "likes": info.likes or 0,
        "downloads": info.downloads or 0,
        "created_at": info.created_at.isoformat() if info.created_at else "",
        "last_modified": info.last_modified.isoformat() if info.last_modified else "",
        "card_data": {
            "language": getattr(info.card_data, "language", None)
            if info.card_data
            else None,
            "license": getattr(info.card_data, "license", None)
            if info.card_data
            else None,
            "datasets": getattr(info.card_data, "datasets", None)
            if info.card_data
            else None,
            "tags": getattr(info.card_data, "tags", None) if info.card_data else None,
        },
    }


def fetch_with_retry(api: HfApi, model_id: str) -> tuple[dict | None, str]:
    """Fetch model info with exponential backoff on rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            info = api.model_info(model_id)
            return model_info_to_raw(info), ""
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Too Many Requests" in err_str:
                wait = BASE_BACKOFF * (2**attempt)
                logger.warning("Rate limited on %s, waiting %.0fs...", model_id, wait)
                time.sleep(wait)
                continue
            if "404" in err_str or "Not Found" in err_str:
                return None, "not_found"
            return None, err_str
    return None, "max_retries_exceeded"


def main():
    # Get model_ids from network DB
    net_conn = sqlite3.connect(str(NETWORK_DB))
    rows = net_conn.execute("SELECT model_id FROM models ORDER BY model_id").fetchall()
    all_ids = [r[0] for r in rows]
    net_conn.close()
    logger.info("Found %d models in network.db", len(all_ids))

    # Init ingest DB
    ing_conn = sqlite3.connect(str(INGEST_DB))
    ing_conn.execute("""
        CREATE TABLE IF NOT EXISTS ingest_models (
            model_id         TEXT PRIMARY KEY,
            source           TEXT DEFAULT 'huggingface',
            likes            INTEGER DEFAULT 0,
            phase_a_done     INTEGER DEFAULT 0,
            phase_b_done     INTEGER DEFAULT 0,
            phase_c_done     INTEGER DEFAULT 0,
            phase_c_attempts INTEGER DEFAULT 0,
            raw_json         TEXT,
            fetched_at       TEXT,
            extracted_at     TEXT,
            vibed_at         TEXT
        )
    """)
    ing_conn.commit()

    # Check which models already have raw_json
    existing = set()
    for r in ing_conn.execute(
        "SELECT model_id FROM ingest_models WHERE raw_json IS NOT NULL"
    ).fetchall():
        existing.add(r[0])
    remaining = [mid for mid in all_ids if mid not in existing]
    logger.info("Already have %d, need to fetch %d", len(existing), len(remaining))

    if not remaining:
        logger.info("Nothing to do!")
        return

    api = HfApi()
    fetched = 0
    not_found = 0
    errors = 0
    start = time.time()
    num_remaining = len(remaining)

    for idx, model_id in enumerate(remaining):
        raw, err = fetch_with_retry(api, model_id)

        if raw is not None:
            ing_conn.execute(
                """INSERT OR REPLACE INTO ingest_models
                   (model_id, source, raw_json, fetched_at)
                   VALUES (?, 'huggingface', ?, datetime('now'))""",
                (model_id, json.dumps(raw)),
            )
            fetched += 1
        elif err == "not_found":
            # Model deleted/renamed — record so we don't retry
            ing_conn.execute(
                """INSERT OR REPLACE INTO ingest_models
                   (model_id, source, raw_json, fetched_at)
                   VALUES (?, 'huggingface', '{"_deleted": true}', datetime('now'))""",
                (model_id,),
            )
            not_found += 1
        else:
            errors += 1
            if errors <= 5:
                logger.warning("Error fetching %s: %s", model_id, err)

        total = idx + 1
        if total % BATCH_COMMIT == 0:
            ing_conn.commit()
            elapsed = time.time() - start
            rate = total / elapsed if elapsed > 0 else 0
            eta = (num_remaining - total) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f/s, ETA %.0fm) fetched=%d not_found=%d errors=%d",
                total,
                num_remaining,
                rate,
                eta / 60,
                fetched,
                not_found,
                errors,
            )

        # Gentle throttle: ~10 req/s to stay under limits
        time.sleep(0.1)

    ing_conn.commit()
    ing_conn.close()
    elapsed = time.time() - start
    logger.info(
        "Done: fetched=%d not_found=%d errors=%d in %.0fs",
        fetched,
        not_found,
        errors,
        elapsed,
    )


if __name__ == "__main__":
    main()
