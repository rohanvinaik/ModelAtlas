"""Retry errored entries in gemini_validation.jsonl.

Reads the JSONL, finds error entries, re-validates them, and writes
a clean JSONL with errors replaced by fresh results.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

from huggingface_hub import HfApi

from gemini_validate import (
    NETWORK_DB,
    INGEST_DB,
    ModelRotator,
    call_gemini,
    get_anchor_dictionary,
    get_our_classification,
    fetch_hf_metadata,
    build_record,
    build_validation_prompt,
    parse_gemini_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _retry_one_entry(conn, api, ingest_conn, rotator, entry, cap_labels, dom_labels):
    """Retry validation for a single errored entry. Returns new record or None."""
    model_id = entry["model_id"]
    downloads = entry.get("downloads", 0)

    ours = get_our_classification(conn, model_id)
    if not ours["anchors"]:
        logger.info("  Skipping — no anchors")
        return None

    hf = fetch_hf_metadata(api, model_id, ingest_conn)
    if not hf:
        logger.info("  Skipping — HF fetch failed")
        return None

    prompt = build_validation_prompt(model_id, ours, hf, cap_labels, dom_labels)
    response = call_gemini(prompt, model=rotator.current)
    parsed = parse_gemini_json(response)
    logger.info("  Fixed -> %s", parsed.get("verdict"))
    return build_record(model_id, downloads, ours, hf, rotator.current, parsed)


def _load_entries(path):
    """Load JSONL entries and find error indices."""
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    error_indices = [i for i, e in enumerate(entries) if e.get("verdict") == "error"]
    return entries, error_indices


def main():
    validation_path = Path.home() / ".cache" / "model-atlas" / "gemini_validation.jsonl"
    entries, error_indices = _load_entries(validation_path)
    logger.info("Found %d errors out of %d entries", len(error_indices), len(entries))

    if not error_indices:
        return

    conn = sqlite3.connect(str(NETWORK_DB))
    conn.row_factory = sqlite3.Row
    ingest_conn = sqlite3.connect(str(INGEST_DB)) if INGEST_DB.exists() else None
    api = HfApi()
    rotator = ModelRotator(models=["gemini-2.5-flash"], interval=float("inf"))

    anchor_dict = get_anchor_dictionary(conn)
    cap_labels = ", ".join(anchor_dict.get("CAPABILITY", []))
    dom_labels = ", ".join(anchor_dict.get("DOMAIN", []))

    fixed = 0
    still_errored = 0
    num_errors = len(error_indices)

    for count, idx in enumerate(error_indices):
        logger.info(
            "[%d/%d] Retrying %s", count + 1, num_errors, entries[idx]["model_id"]
        )
        try:
            result = _retry_one_entry(
                conn, api, ingest_conn, rotator, entries[idx], cap_labels, dom_labels
            )
            if result:
                entries[idx] = result
                fixed += 1
            else:
                still_errored += 1
        except Exception as e:
            logger.warning("  Still errored: %s", e)
            still_errored += 1
        time.sleep(0.5)

    conn.close()
    if ingest_conn:
        ingest_conn.close()

    with open(validation_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    logger.info("Done. Fixed %d, still errored %d", fixed, still_errored)


if __name__ == "__main__":
    main()
