"""Background ingestion daemon — Phase A (fetch), Phase B (extract).

Phase C vibe extraction in ingest_vibes.py. CLI dispatch in ingest_cli.py.
"""

from __future__ import annotations

import json
import logging
import signal
import sqlite3
import time
from datetime import datetime, timezone

from . import db, db_ingest
from .config import (
    INGEST_BATCH_SIZE,
    INGEST_MIN_LIKES,
    INGEST_VIBE_MIN_LIKES,
)
from .extraction.deterministic import ModelInput
from .extraction.pipeline import extract_and_store

logger = logging.getLogger(__name__)

# Graceful shutdown flag
_shutdown = False


def _handle_signal(signum: int, frame: object) -> None:
    global _shutdown
    logger.info("Received signal %d, finishing current batch...", signum)
    _shutdown = True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Phase A: Fetch metadata from all sources ---


def _phase_a_huggingface(
    ingest_conn: sqlite3.Connection,
    min_likes: int = INGEST_MIN_LIKES,
) -> int:
    """Stream models from HuggingFace and cache raw JSON."""
    from huggingface_hub import HfApi

    api = HfApi()
    count = 0
    logger.info("Phase A [huggingface]: streaming models (min_likes=%d)...", min_likes)

    for model in api.list_models(full=True, cardData=True, sort="likes", direction=-1):
        if _shutdown:
            break

        likes = model.likes or 0
        if likes < min_likes:
            break

        model_id = model.id or ""
        if not model_id:
            continue

        # Serialize model to JSON (INSERT OR IGNORE handles duplicates)
        raw = {
            "model_id": model_id,
            "author": model.author or "",
            "pipeline_tag": model.pipeline_tag or "",
            "tags": list(model.tags or []),
            "library_name": model.library_name or "",
            "likes": likes,
            "downloads": model.downloads or 0,
            "created_at": str(model.created_at) if model.created_at else None,
            "license": getattr(model, "license", "") or "",
            "safetensors_info": (
                _safetensors_to_dict(model.safetensors)
                if hasattr(model, "safetensors") and model.safetensors
                else None
            ),
            "source": "huggingface",
        }

        ingest_conn.execute(
            """INSERT OR IGNORE INTO ingest_models
               (model_id, source, likes, phase_a_done, raw_json, fetched_at)
               VALUES (?, 'huggingface', ?, 1, ?, ?)""",
            (model_id, likes, json.dumps(raw), _now_iso()),
        )
        count += 1

        if count % INGEST_BATCH_SIZE == 0:
            ingest_conn.commit()
            logger.info("Phase A [huggingface]: %d models fetched...", count)

    ingest_conn.commit()
    logger.info("Phase A [huggingface]: complete — %d new models", count)
    return count


def _safetensors_to_dict(info: object) -> dict | None:
    """Convert safetensors info object to a serializable dict."""
    if info is None:
        return None
    if isinstance(info, dict):
        return info
    # huggingface_hub SafetensorsInfo object
    result = {}
    if hasattr(info, "parameters"):
        result["parameters"] = info.parameters
    if hasattr(info, "total"):
        result["total"] = info.total
    return result if result else None


def _phase_a_ollama(ingest_conn: sqlite3.Connection) -> int:
    """Fetch models from local Ollama instance."""
    import urllib.error
    import urllib.request

    count = 0
    logger.info("Phase A [ollama]: fetching local models...")

    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError) as e:
        logger.info("Phase A [ollama]: not available (%s), skipping", e)
        return 0

    for m in data.get("models", []):
        if _shutdown:
            break

        name = m.get("name", "")
        if not name:
            continue

        model_id = f"ollama/{name}"

        # Fetch details (INSERT OR IGNORE handles duplicates)
        details = {}
        try:
            body = json.dumps({"name": name}).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/show",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                details = json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError):
            pass

        model_details = details.get("details", {})
        raw = {
            "model_id": model_id,
            "author": "",
            "pipeline_tag": "",
            "tags": _ollama_tags(model_details),
            "library_name": "",
            "likes": 0,
            "downloads": 0,
            "created_at": m.get("modified_at"),
            "source": "ollama",
            "ollama_details": model_details,
        }

        ingest_conn.execute(
            """INSERT OR IGNORE INTO ingest_models
               (model_id, source, likes, phase_a_done, raw_json, fetched_at)
               VALUES (?, 'ollama', 0, 1, ?, ?)""",
            (model_id, json.dumps(raw), _now_iso()),
        )
        count += 1

    ingest_conn.commit()
    logger.info("Phase A [ollama]: complete — %d new models", count)
    return count


def _ollama_tags(details: dict) -> list[str]:
    """Build synthetic tags from Ollama model details."""
    tags: list[str] = []
    family = details.get("family", "")
    if family:
        tags.append(family.lower())
    quant = details.get("quantization_level", "")
    if quant:
        tags.append(quant.lower())
        tags.append("quantized")
    tags.extend(["GGUF-available", "llama-cpp-compatible", "CPU-inference"])
    return tags


def phase_a(
    ingest_conn: sqlite3.Connection,
    min_likes: int = INGEST_MIN_LIKES,
) -> dict[str, int]:
    """Run Phase A across all available sources."""
    results: dict[str, int] = {}
    results["huggingface"] = _phase_a_huggingface(ingest_conn, min_likes)
    if not _shutdown:
        results["ollama"] = _phase_a_ollama(ingest_conn)
    return results


# --- Phase B: Tier 1+2 extraction ---


def phase_b(
    ingest_conn: sqlite3.Connection,
    network_conn: sqlite3.Connection,
) -> int:
    """Run Tier 1+2 extraction on models with phase_a_done=1, phase_b_done=0."""
    cursor = ingest_conn.execute(
        """SELECT model_id, raw_json FROM ingest_models
           WHERE phase_a_done = 1 AND phase_b_done = 0
           ORDER BY likes DESC"""
    )

    count = 0
    for row in cursor:
        if _shutdown:
            break

        model_id = row["model_id"]
        try:
            raw = json.loads(row["raw_json"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Phase B: bad JSON for %s, skipping", model_id)
            continue

        try:
            inp = ModelInput(
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
            extract_and_store(network_conn, inp)

            ingest_conn.execute(
                """UPDATE ingest_models
                   SET phase_b_done = 1, extracted_at = ?
                   WHERE model_id = ?""",
                (_now_iso(), model_id),
            )
            count += 1

            if count % INGEST_BATCH_SIZE == 0:
                network_conn.commit()
                ingest_conn.commit()
                logger.info("Phase B: %d models extracted...", count)

        except Exception:
            logger.warning("Phase B: failed %s", model_id, exc_info=True)

    network_conn.commit()
    ingest_conn.commit()
    logger.info("Phase B: complete — %d models extracted", count)
    return count


# --- Status ---


def get_status(ingest_conn: sqlite3.Connection) -> dict:
    """Get ingest progress summary."""
    row = ingest_conn.execute(
        "SELECT COUNT(*) as total, SUM(phase_a_done) as a, SUM(phase_b_done) as b, SUM(phase_c_done) as c FROM ingest_models"
    ).fetchone()
    total = row["total"]
    phase_a_done = row["a"] or 0
    phase_b_done = row["b"] or 0
    phase_c_done = row["c"] or 0

    # Per-source breakdown
    sources = {}
    for row in ingest_conn.execute(
        """SELECT source,
                  COUNT(*) as total,
                  SUM(phase_a_done) as a_done,
                  SUM(phase_b_done) as b_done,
                  SUM(phase_c_done) as c_done
           FROM ingest_models GROUP BY source"""
    ).fetchall():
        sources[row["source"]] = {
            "total": row["total"],
            "phase_a": row["a_done"],
            "phase_b": row["b_done"],
            "phase_c": row["c_done"],
        }

    return {
        "total_models": total,
        "phase_a_done": phase_a_done,
        "phase_b_done": phase_b_done,
        "phase_c_done": phase_c_done,
        "phase_b_pending": phase_a_done - phase_b_done,
        "phase_c_pending": phase_b_done - phase_c_done,
        "by_source": sources,
    }


def print_status(ingest_conn: sqlite3.Connection) -> None:
    """Print human-readable ingest status."""
    status = get_status(ingest_conn)
    print("ModelAtlas Ingest Status")
    print(f"{'=' * 40}")
    print(f"Total models tracked: {status['total_models']}")
    print(f"Phase A (fetched):    {status['phase_a_done']}")
    print(f"Phase B (extracted):  {status['phase_b_done']}")
    print(f"Phase C (vibed):      {status['phase_c_done']}")
    print(f"Phase B pending:      {status['phase_b_pending']}")
    print(f"Phase C pending:      {status['phase_c_pending']}")
    print()
    for source, info in status["by_source"].items():
        print(
            f"  [{source}] total={info['total']} A={info['phase_a']} B={info['phase_b']} C={info['phase_c']}"
        )


# --- Pipeline orchestrator ---


def run(
    phases: str = "abc",
    min_likes: int = INGEST_MIN_LIKES,
    vibe_min_likes: int = INGEST_VIBE_MIN_LIKES,
    daemon: bool = False,
    daemon_sleep: int = 86400,
) -> None:
    """Run the ingest pipeline."""
    valid_phases = set("abc")
    invalid = set(phases) - valid_phases
    if invalid:
        raise ValueError(f"Invalid phase(s): {invalid!r}. Valid phases: a, b, c")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    ingest_conn = db_ingest.get_connection()
    db_ingest.init_db(ingest_conn)

    network_conn = db.get_connection()
    db.init_db(network_conn)

    global _shutdown

    while True:
        _shutdown = False

        if "a" in phases and not _shutdown:
            phase_a(ingest_conn, min_likes=min_likes)

        if "b" in phases and not _shutdown:
            phase_b(ingest_conn, network_conn)

        if "c" in phases and not _shutdown:
            from .ingest_vibes import phase_c

            phase_c(
                ingest_conn,
                network_conn,
                vibe_min_likes=vibe_min_likes,
                is_shutdown=lambda: _shutdown,
            )

        if not daemon or _shutdown:
            break

        logger.info("Daemon sleeping for %d seconds...", daemon_sleep)
        for _ in range(daemon_sleep):
            if _shutdown:
                break
            time.sleep(1)

    ingest_conn.close()
    network_conn.close()
    logger.info("Ingest complete")
