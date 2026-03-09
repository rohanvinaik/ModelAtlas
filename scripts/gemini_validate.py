"""Gemini-powered validation of C2 anchor classifications.

Prioritizes top-download models. For each, fetches raw HF metadata,
compares against our classification, and asks Gemini to diff-analyze.

Usage:
    python scripts/gemini_validate.py [--limit 20] [--offset 0] [--output results.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "model-atlas"
NETWORK_DB = CACHE_DIR / "network.db"
INGEST_DB = CACHE_DIR / "ingest_state.db"

VALIDATION_PROMPT = """You are auditing an ML model classification system. Below is the RAW metadata from HuggingFace for a model, and what our automated system classified it as.

## Raw HuggingFace Metadata
- Model ID: {model_id}
- Author: {author}
- Pipeline tag: {pipeline_tag}
- Tags: {tags}
- Library: {library_name}
- Downloads: {downloads:,}
- Likes: {likes:,}

## Our System's Classification
- Summary: {our_summary}
- Assigned anchors: {our_anchors}

## Full anchor dictionary (what the system could have chosen from)
CAPABILITY: {cap_anchors}
DOMAIN: {dom_anchors}

## Task
Analyze the accuracy of our classification against the raw metadata. Respond with JSON only:
{{
  "verdict": "correct" | "partially_correct" | "wrong",
  "correct_anchors": ["anchors that were correctly assigned"],
  "wrong_anchors": ["anchors that should NOT have been assigned"],
  "missing_anchors": ["anchors that SHOULD have been assigned but weren't"],
  "suggested_summary": "A better one-sentence summary if ours is wrong, or null if acceptable",
  "reasoning": "Brief explanation of what's right and wrong"
}}"""


def get_top_models(
    conn: sqlite3.Connection, limit: int, offset: int
) -> list[tuple[str, int]]:
    """Get top models by downloads that have C2 anchors."""
    rows = conn.execute(
        """SELECT DISTINCT ma.model_id, CAST(COALESCE(mm.value, '0') AS INTEGER) as dl
           FROM model_anchors ma
           LEFT JOIN model_metadata mm ON ma.model_id = mm.model_id AND mm.key = 'downloads'
           WHERE ma.confidence = 0.5
           ORDER BY dl DESC
           LIMIT ? OFFSET ?""",
        (limit, offset),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_our_classification(conn: sqlite3.Connection, model_id: str) -> dict:
    """Get our C2 anchors and summary for a model."""
    anchors = conn.execute(
        """SELECT a.label, a.bank FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ? AND ma.confidence = 0.5
           ORDER BY a.bank, a.label""",
        (model_id,),
    ).fetchall()

    summary_row = conn.execute(
        "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'qwen_summary'",
        (model_id,),
    ).fetchone()

    return {
        "anchors": [{"label": r[0], "bank": r[1]} for r in anchors],
        "summary": summary_row[0] if summary_row else "",
    }


def get_anchor_dictionary(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Get all anchor labels by bank."""
    rows = conn.execute(
        "SELECT label, bank FROM anchors ORDER BY bank, label"
    ).fetchall()
    by_bank: dict[str, list[str]] = {}
    for label, bank in rows:
        by_bank.setdefault(bank, []).append(label)
    return by_bank


def fetch_hf_metadata(
    api: HfApi, model_id: str, ingest_conn: sqlite3.Connection | None = None
) -> dict:
    """Fetch raw metadata, preferring cached raw_json from ingest_state.db."""
    # Try local cache first
    if ingest_conn is not None:
        row = ingest_conn.execute(
            "SELECT raw_json FROM ingest_models WHERE model_id = ? AND raw_json IS NOT NULL",
            (model_id,),
        ).fetchone()
        if row and row[0] and "_deleted" not in row[0]:
            try:
                raw = json.loads(row[0])
                return {
                    "author": raw.get("author", ""),
                    "pipeline_tag": raw.get("pipeline_tag", ""),
                    "tags": raw.get("tags", []),
                    "library_name": raw.get("library_name", ""),
                    "downloads": raw.get("downloads", 0),
                    "likes": raw.get("likes", 0),
                }
            except (json.JSONDecodeError, TypeError):
                pass

    # Fall back to HF API
    try:
        info = api.model_info(model_id)
        return {
            "author": info.author or "",
            "pipeline_tag": info.pipeline_tag or "",
            "tags": list(info.tags or []),
            "library_name": info.library_name or "",
            "downloads": info.downloads or 0,
            "likes": info.likes or 0,
        }
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", model_id, e)
        return {}


MODELS_ROTATION = ["gemini-2.5-pro", "gemini-2.5-flash"]
ROTATION_INTERVAL = 30 * 60  # 30 minutes


class ModelRotator:
    """Alternates between Gemini models on a timer to spread quota usage."""

    def __init__(
        self, models: list[str] = MODELS_ROTATION, interval: float = ROTATION_INTERVAL
    ):
        self._models = models
        self._interval = interval
        self._index = 0
        self._switch_at = time.monotonic() + interval

    @property
    def current(self) -> str:
        now = time.monotonic()
        if now >= self._switch_at:
            self._index = (self._index + 1) % len(self._models)
            self._switch_at = now + self._interval
            logger.info("Rotating model -> %s", self._models[self._index])
        return self._models[self._index]


def call_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Call Gemini CLI with a prompt, return response."""
    result = subprocess.run(
        ["gemini", "-m", model, "-p", prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout.strip()


def parse_gemini_json(response: str) -> dict:
    """Extract and parse JSON from a Gemini response (handles markdown blocks)."""
    clean = response
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0]
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0]
    return json.loads(clean.strip())


def build_validation_prompt(
    model_id: str,
    ours: dict,
    hf: dict,
    cap_labels: str,
    dom_labels: str,
) -> str:
    """Build the Gemini validation prompt for a single model."""
    anchor_labels = [a["label"] for a in ours["anchors"]]
    return VALIDATION_PROMPT.format(
        model_id=model_id,
        author=hf.get("author", ""),
        pipeline_tag=hf.get("pipeline_tag", ""),
        tags=", ".join(hf.get("tags", [])[:20]),
        library_name=hf.get("library_name", ""),
        downloads=hf.get("downloads", 0),
        likes=hf.get("likes", 0),
        our_summary=ours["summary"],
        our_anchors=", ".join(anchor_labels),
        cap_anchors=cap_labels,
        dom_anchors=dom_labels,
    )


def build_record(
    model_id: str, downloads: int, ours: dict, hf: dict, gemini_model: str, parsed: dict
) -> dict:
    """Build an output record from validation results."""
    return {
        "model_id": model_id,
        "downloads": downloads,
        "our_anchors": [a["label"] for a in ours["anchors"]],
        "our_summary": ours["summary"],
        "hf_pipeline_tag": hf.get("pipeline_tag", ""),
        "hf_tags": hf.get("tags", [])[:20],
        "gemini_model": gemini_model,
        **parsed,
    }


def _validate_one_model(
    conn, api, ingest_conn, rotator, model_id, downloads, cap_labels, dom_labels, out
):
    """Validate a single model. Returns (verdict, consecutive_error_reset)."""
    ours = get_our_classification(conn, model_id)
    if not ours["anchors"]:
        logger.info("  Skipping — no C2 anchors")
        return None, False

    hf = fetch_hf_metadata(api, model_id, ingest_conn)
    if not hf:
        logger.info("  Skipping — HF fetch failed")
        return None, False

    prompt = build_validation_prompt(model_id, ours, hf, cap_labels, dom_labels)
    response = call_gemini(prompt, model=rotator.current)
    parsed = parse_gemini_json(response)
    verdict = parsed.get("verdict", "error")

    record = build_record(model_id, downloads, ours, hf, rotator.current, parsed)
    out.write(json.dumps(record) + "\n")
    out.flush()

    logger.info(
        "  Verdict: %s | wrong=%s missing=%s",
        verdict,
        parsed.get("wrong_anchors", []),
        parsed.get("missing_anchors", []),
    )
    return verdict, True


def main():
    parser = argparse.ArgumentParser(
        description="Validate C2 classifications with Gemini"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Number of models to validate"
    )
    parser.add_argument("--offset", type=int, default=0, help="Skip first N models")
    parser.add_argument(
        "--output", default="validation_results.jsonl", help="Output JSONL file"
    )
    parser.add_argument(
        "--model", default=None, help="Force a single model (disables rotation)"
    )
    parser.add_argument(
        "--rotate-interval",
        type=int,
        default=30,
        help="Rotation interval in minutes (default: 30)",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(str(NETWORK_DB))
    conn.row_factory = sqlite3.Row
    ingest_conn = sqlite3.connect(str(INGEST_DB)) if INGEST_DB.exists() else None
    api = HfApi()

    if args.model:
        rotator = ModelRotator(models=[args.model], interval=float("inf"))
    else:
        rotator = ModelRotator(interval=args.rotate_interval * 60)

    models = get_top_models(conn, args.limit, args.offset)
    anchor_dict = get_anchor_dictionary(conn)
    cap_labels = ", ".join(anchor_dict.get("CAPABILITY", []))
    dom_labels = ", ".join(anchor_dict.get("DOMAIN", []))

    logger.info(
        "Validating %d models (offset=%d), starting with %s",
        len(models),
        args.offset,
        rotator.current,
    )

    results = {"correct": 0, "partially_correct": 0, "wrong": 0, "error": 0}
    consecutive_errors = 0
    num_models = len(models)

    with open(Path(args.output), "a") as out:
        for idx, (model_id, downloads) in enumerate(models):
            logger.info(
                "[%d/%d] %s (%s downloads)",
                idx + 1,
                num_models,
                model_id,
                f"{downloads:,}",
            )

            try:
                verdict, ok = _validate_one_model(
                    conn,
                    api,
                    ingest_conn,
                    rotator,
                    model_id,
                    downloads,
                    cap_labels,
                    dom_labels,
                    out,
                )
                if ok:
                    results[verdict] = results.get(verdict, 0) + 1
                    consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                results["error"] += 1
                logger.warning(
                    "  Gemini error (%d consecutive): %s", consecutive_errors, e
                )
                if consecutive_errors >= 5:
                    logger.error(
                        "5 consecutive errors — likely quota exhaustion. Stopping."
                    )
                    break

            time.sleep(0.5)

    conn.close()
    if ingest_conn is not None:
        ingest_conn.close()

    logger.info("Results: %s", results)
    total = sum(results.values())
    if total > 0:
        logger.info(
            "Accuracy: %.1f%% correct, %.1f%% partial, %.1f%% wrong",
            100 * results.get("correct", 0) / total,
            100 * results.get("partially_correct", 0) / total,
            100 * results.get("wrong", 0) / total,
        )


if __name__ == "__main__":
    main()
