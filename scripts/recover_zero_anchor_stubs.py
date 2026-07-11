#!/usr/bin/env python3
"""One-shot recovery for zero-anchor stub models.

Background: when Phase D/E lineage analysis or Phase E web enrichment
mentions a model that's not in the catalog, a stub row is inserted into
`models` with source='stub'. Some stubs accumulate zero anchors / positions
because (a) Phase A never fetched their HF metadata, or (b) Phase A succeeded
but Phase B's extractor returned nothing usable.

This script targets the second-class citizens: rows in `models` with
source='stub' that have zero rows in `model_anchors`. For each:

- If `ingest_state.ingest_models` already has raw_json: re-run extraction
  (cheap; useful when extractor logic has improved since the original Phase B).
- Else: fetch metadata + config from HuggingFace, then extract. Skip models
  that 404 (deleted, private, autotrain-ephemeral).

Run with --dry-run to preview, then again without it to apply.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from model_atlas import db
from model_atlas.config import CACHE_DIR
from model_atlas.extraction.pipeline import extract_and_store
from model_atlas.sources.base import ModelInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("recover")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _zero_anchor_stubs(network: sqlite3.Connection) -> list[str]:
    cur = network.execute(
        """SELECT m.model_id FROM models m
           WHERE m.source = 'stub'
             AND NOT EXISTS (SELECT 1 FROM model_anchors WHERE model_id = m.model_id)"""
    )
    return [r[0] for r in cur.fetchall()]


def _input_from_raw(model_id: str, raw: dict) -> ModelInput:
    return ModelInput(
        model_id=raw.get("model_id", model_id),
        author=raw.get("author", ""),
        pipeline_tag=raw.get("pipeline_tag", ""),
        tags=raw.get("tags", []),
        library_name=raw.get("library_name", ""),
        likes=raw.get("likes", 0),
        downloads=raw.get("downloads", 0),
        created_at=raw.get("created_at"),
        license_str=raw.get("license", ""),
        safetensors_info=raw.get("safetensors_info"),
        config=raw.get("config"),
        source=raw.get("source", "huggingface"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no writes.")
    parser.add_argument("--limit", type=int, default=None, help="Cap how many to process (for smoke-test).")
    parser.add_argument("--fetch-config", action="store_true", default=True,
                        help="Fetch config.json from HF for ARCHITECTURE classification (default: on).")
    parser.add_argument("--skip-config", dest="fetch_config", action="store_false",
                        help="Skip config.json fetch (faster but worse ARCHITECTURE accuracy).")
    args = parser.parse_args()

    network = db.get_connection()
    db.init_db(network)
    ingest_path = Path(CACHE_DIR) / "ingest_state.db"
    ingest = sqlite3.connect(ingest_path)
    ingest.row_factory = sqlite3.Row

    stubs = _zero_anchor_stubs(network)
    if args.limit:
        stubs = stubs[: args.limit]
    log.info("zero-anchor stubs found: %d", len(stubs))

    # Bucket: cached raw vs needs fetch
    placeholders = ",".join("?" * len(stubs))
    cached = {
        r["model_id"]: json.loads(r["raw_json"])
        for r in ingest.execute(
            f"SELECT model_id, raw_json FROM ingest_models "
            f"WHERE model_id IN ({placeholders}) AND raw_json IS NOT NULL",
            stubs,
        )
    }
    need_fetch = [m for m in stubs if m not in cached]
    log.info("  with cached raw_json: %d (re-extract)", len(cached))
    log.info("  need HF fetch:       %d", len(need_fetch))

    if args.dry_run:
        log.info("DRY RUN — would re-extract %d cached + fetch+extract %d remote", len(cached), len(need_fetch))
        return 0

    # Phase 1: re-extract from cached raw_json
    re_extract_ok = re_extract_empty = 0
    for model_id, raw in cached.items():
        try:
            inp = _input_from_raw(model_id, raw)
            extract_and_store(network, inp)
            # Confirm anchors landed
            n = network.execute(
                "SELECT COUNT(*) FROM model_anchors WHERE model_id=?", (model_id,)
            ).fetchone()[0]
            if n > 0:
                re_extract_ok += 1
            else:
                re_extract_empty += 1
        except Exception as e:
            log.warning("re-extract failed for %s: %s", model_id, e)
    network.commit()
    log.info("re-extract: %d gained anchors, %d still empty", re_extract_ok, re_extract_empty)

    # Phase 2: fetch + extract from HF
    if not need_fetch:
        return 0

    from huggingface_hub.errors import (
        EntryNotFoundError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    from model_atlas.sources.huggingface import HuggingFaceAdapter

    hf = HuggingFaceAdapter()
    fetched_ok = fetched_404 = fetched_err = extracted_ok = extracted_empty = 0

    for i, model_id in enumerate(need_fetch, 1):
        if i % 25 == 0:
            log.info("  fetch progress: %d / %d", i, len(need_fetch))
        try:
            inp = hf.get_detail(model_id)
            fetched_ok += 1
        except (RepositoryNotFoundError, EntryNotFoundError):
            fetched_404 += 1
            continue
        except HfHubHTTPError as e:
            fetched_err += 1
            log.debug("HF http error %s: %s", model_id, e)
            continue
        except Exception as e:
            fetched_err += 1
            log.debug("unexpected fetch error %s: %s", model_id, e)
            continue

        try:
            extract_and_store(network, inp)
            # Stamp ingest_state so future scans don't re-fetch
            ingest.execute(
                """INSERT INTO ingest_models (model_id, source, likes, phase_a_done, phase_b_done, raw_json, fetched_at, extracted_at)
                   VALUES (?, 'huggingface', ?, 1, 1, ?, ?, ?)
                   ON CONFLICT(model_id) DO UPDATE SET
                     phase_a_done=1, phase_b_done=1, raw_json=excluded.raw_json,
                     fetched_at=excluded.fetched_at, extracted_at=excluded.extracted_at""",
                (
                    model_id,
                    inp.likes,
                    json.dumps({
                        "model_id": inp.model_id,
                        "author": inp.author,
                        "pipeline_tag": inp.pipeline_tag,
                        "tags": inp.tags,
                        "library_name": inp.library_name,
                        "likes": inp.likes,
                        "downloads": inp.downloads,
                        "created_at": inp.created_at,
                        "license": inp.license_str,
                        "safetensors_info": inp.safetensors_info,
                        "config": inp.config,
                        "source": "huggingface",
                    }),
                    _now_iso(),
                    _now_iso(),
                ),
            )
            n = network.execute(
                "SELECT COUNT(*) FROM model_anchors WHERE model_id=?", (model_id,)
            ).fetchone()[0]
            if n > 0:
                extracted_ok += 1
            else:
                extracted_empty += 1
        except Exception as e:
            log.warning("extract failed %s: %s", model_id, e)

        if i % 50 == 0:
            network.commit()
            ingest.commit()

    network.commit()
    ingest.commit()
    log.info("HF fetch: %d ok, %d 404, %d error", fetched_ok, fetched_404, fetched_err)
    log.info("HF extract: %d gained anchors, %d still empty", extracted_ok, extracted_empty)

    # Bump source from 'stub' to 'huggingface' ONLY for models this run successfully
    # fetched from HF and produced anchors for. Scoped to `need_fetch` (the input
    # bucket for HF fetches) to avoid sweeping in pre-existing stub rows that
    # happen to have anchors from unrelated code paths.
    promoted = 0
    if need_fetch:
        placeholders = ",".join("?" * len(need_fetch))
        network.execute(
            f"""UPDATE models SET source='huggingface'
                WHERE source='stub'
                  AND model_id IN ({placeholders})
                  AND EXISTS (SELECT 1 FROM model_anchors WHERE model_id = models.model_id)""",
            need_fetch,
        )
        promoted = network.execute("SELECT changes()").fetchone()[0]
        network.commit()
    log.info("promoted source: 'stub' → 'huggingface' for %d models (scoped to this run)", promoted)

    return 0


if __name__ == "__main__":
    sys.exit(main())
