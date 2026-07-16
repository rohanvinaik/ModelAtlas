#!/usr/bin/env python3
"""Targeted re-pull + re-extract for models the certifier flags as noisy.

Path B of the audit-cleanup overhaul: pull fresh HF data for every model
that the current rule set would REJECT any anchor on, wipe that model's
existing anchor rows, and re-run the extraction pipeline. Since
extract_and_store now routes through the certifier at write time, the
result is a clean anchor set per model — sourced from HF, filtered by
structural rules, no leftover legacy noise.

Sequence per model:
  1. Fetch fresh HF metadata via HuggingFaceAdapter.get_detail
  2. Update ingest_state.raw_json + fetched_at so subsequent cache reads
     see the fresh data
  3. DELETE all existing model_anchors for this model_id
  4. Call extract_and_store — deterministic + pattern anchors routed
     through the certifier before writing

--dry-run walks the certifier once, lists the model_ids that would be
re-pulled, and stops. Default behavior is to require --apply.

Snapshot the DB before --apply. This script deletes rows.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter

from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from model_atlas import db
from model_atlas.certifier import HFFacts, certify
from model_atlas.config import CACHE_DIR
from model_atlas.contract import (
    AnchorEmission,
    Bank,
    CertificationOutcome,
    EvidenceType,
    Provenance,
)
from model_atlas.extraction.pipeline import extract_and_store
from model_atlas.sources.base import ModelInput
from model_atlas.sources.huggingface import HuggingFaceAdapter


def _facts_from_ingest(ingest: sqlite3.Connection, model_id: str) -> HFFacts | None:
    row = ingest.execute(
        "SELECT raw_json FROM ingest_models WHERE model_id = ?", (model_id,)
    ).fetchone()
    if not row or not row[0]:
        return None
    raw = json.loads(row[0])
    return HFFacts(
        model_id=model_id,
        pipeline_tag=str(raw.get("pipeline_tag") or ""),
        library_name=str(raw.get("library_name") or ""),
        license=str(raw.get("license") or ""),
        tags=tuple(str(t) for t in (raw.get("tags") or [])),
        model_type=str((raw.get("config") or {}).get("model_type") or ""),
        safetensors_present=bool(raw.get("safetensors_info")),
        config=dict(raw.get("config") or {}),
    )


def _live_vocab(conn: sqlite3.Connection) -> dict[str, Bank]:
    out: dict[str, Bank] = {}
    for label, bank_str in conn.execute("SELECT label, bank FROM anchors"):
        try:
            out[label] = Bank(bank_str)
        except ValueError:
            continue
    return out


def _current_emissions(
    network: sqlite3.Connection, model_id: str, live_vocab: dict[str, Bank]
) -> list[AnchorEmission]:
    rows = network.execute(
        """SELECT a.label, a.bank, ma.confidence
           FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ?""",
        (model_id,),
    ).fetchall()
    ems: list[AnchorEmission] = []
    for label, bank_str, conf in rows:
        bank = live_vocab.get(label)
        if bank is None:
            try:
                bank = Bank(bank_str)
            except ValueError:
                continue
        try:
            ems.append(AnchorEmission(
                model_id=model_id, label=label, bank=bank, confidence=float(conf),
                evidence=Provenance(EvidenceType.LLM_INFERENCE, "current_state", "repull_scan"),
            ))
        except ValueError:
            continue
    return ems


def identify_flagged(
    network: sqlite3.Connection, ingest: sqlite3.Connection
) -> list[str]:
    """Return the model_ids that certify() would REJECT at least one anchor on."""
    live_vocab = _live_vocab(network)
    flagged: list[str] = []
    total = network.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    print(f"Scanning {total:,} models to identify flagged set...")
    ids = [r[0] for r in network.execute("SELECT model_id FROM models")]
    for i, mid in enumerate(ids, 1):
        if i % 5000 == 0:
            print(f"  scanned {i:,}/{len(ids):,}  flagged so far: {len(flagged):,}")
        facts = _facts_from_ingest(ingest, mid)
        if facts is None:
            # No cached raw_json — skip, will be re-pulled separately if desired
            continue
        proposed = _current_emissions(network, mid, live_vocab)
        result = certify(facts, proposed, live_vocab=live_vocab)
        if any(v.outcome is CertificationOutcome.REJECTED for v in result.verdicts):
            flagged.append(mid)
    return flagged


def _upsert_ingest_state(
    ingest: sqlite3.Connection, inp: ModelInput, when_iso: str
) -> None:
    ingest.execute(
        """INSERT INTO ingest_models (model_id, source, likes, phase_a_done, phase_b_done, raw_json, fetched_at, extracted_at)
           VALUES (?, 'huggingface', ?, 1, 1, ?, ?, ?)
           ON CONFLICT(model_id) DO UPDATE SET
             phase_a_done=1, phase_b_done=1, raw_json=excluded.raw_json,
             fetched_at=excluded.fetched_at, extracted_at=excluded.extracted_at""",
        (
            inp.model_id, inp.likes,
            json.dumps({
                "model_id": inp.model_id, "author": inp.author,
                "pipeline_tag": inp.pipeline_tag, "tags": inp.tags,
                "library_name": inp.library_name, "likes": inp.likes,
                "downloads": inp.downloads, "created_at": inp.created_at,
                "license": inp.license_str, "safetensors_info": inp.safetensors_info,
                "config": inp.config, "source": "huggingface",
            }),
            when_iso, when_iso,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="List flagged models and stop (no HF fetch, no DB writes).")
    parser.add_argument("--apply", action="store_true",
                        help="Perform the re-pull + re-extract. Snapshot the DB first.")
    parser.add_argument("--input-list", type=str, default=None,
                        help="File with one model_id per line (skip auto-flagged scan).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of models processed.")
    parser.add_argument("--output-list", type=str, default=None,
                        help="Save flagged model_ids here.")
    args = parser.parse_args()

    if not (args.dry_run or args.apply):
        print("must pass --dry-run or --apply", file=sys.stderr)
        return 2

    network = db.get_connection()
    db.init_db(network)
    ingest = sqlite3.connect(f"{CACHE_DIR}/ingest_state.db")

    # ---- Build the flagged list ----
    if args.input_list:
        with open(args.input_list) as f:
            flagged = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(flagged):,} model_ids from {args.input_list}")
    else:
        flagged = identify_flagged(network, ingest)
        print(f"\nFlagged model_ids: {len(flagged):,}")

    if args.output_list and flagged:
        with open(args.output_list, "w") as f:
            for mid in flagged:
                f.write(mid + "\n")
        print(f"  wrote list to {args.output_list}")

    if args.limit:
        flagged = flagged[: args.limit]
        print(f"  capped at {len(flagged):,} (--limit)")

    if args.dry_run:
        print("(dry-run — pass --apply to perform the re-pull)")
        return 0

    # ---- Re-pull + re-extract ----
    hf = HuggingFaceAdapter()
    stats: Counter[str] = Counter()
    from datetime import datetime, timezone
    when_iso = datetime.now(timezone.utc).isoformat()

    for i, mid in enumerate(flagged, 1):
        if i % 25 == 0:
            print(f"  progress: {i}/{len(flagged)}  fetched={stats['fetched']} "
                  f"404={stats['404']} extract_ok={stats['extract_ok']}")
        try:
            inp = hf.get_detail(mid)
            stats["fetched"] += 1
        except (RepositoryNotFoundError, EntryNotFoundError):
            stats["404"] += 1
            continue
        except HfHubHTTPError:
            stats["http_error"] += 1
            continue
        except Exception:
            stats["fetch_error"] += 1
            continue

        # Wipe existing anchors so re-extract fully rebuilds the certified set
        network.execute("DELETE FROM model_anchors WHERE model_id = ?", (mid,))

        try:
            extract_and_store(network, inp, card_text="")
            stats["extract_ok"] += 1
        except Exception as exc:
            stats["extract_error"] += 1
            print(f"    ERR extract {mid}: {exc}", file=sys.stderr)
            continue

        _upsert_ingest_state(ingest, inp, when_iso)

        if i % 25 == 0:
            network.commit()
            ingest.commit()
            # Small pause to be nice to HF API
            time.sleep(0.1)

    network.commit()
    ingest.commit()

    print()
    print("=" * 60)
    for k, v in stats.most_common():
        print(f"  {k:20s} {v:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
