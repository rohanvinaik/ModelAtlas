"""Background ingestion daemon for ModelAtlas.

Three-phase pipeline that populates the semantic network:
  Phase A: Fetch metadata from all registered sources (HF, Ollama, etc.)
  Phase B: Run Tier 1+2 extraction (pure Python, no network I/O)
  Phase C: Vibe extraction via Outlines + local LLM (structured output)

Usage:
  python -m model_atlas.ingest --phase ab               # fetch + extract
  python -m model_atlas.ingest --phase c --vibe-min-likes 500  # vibes only
  python -m model_atlas.ingest --daemon                  # full daemon loop
  python -m model_atlas.ingest --status                  # show progress
"""

from __future__ import annotations

import argparse
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
    VIBE_MAX_RETRIES,
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


# ---------------------------------------------------------------------------
# Phase A: Fetch metadata from all sources
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase B: Tier 1+2 extraction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase C: Vibe extraction via Outlines
# ---------------------------------------------------------------------------


def _store_vibe_result(
    network_conn: sqlite3.Connection,
    model_id: str,
    result: object,
) -> None:
    """Persist vibe extraction results (summary + extra anchors)."""
    if result.summary:  # type: ignore[union-attr]
        db.set_metadata(
            network_conn, model_id, "vibe_summary", result.summary, "str"  # type: ignore[union-attr]
        )
    for anchor_label in result.extra_anchors:  # type: ignore[union-attr]
        anchor_label = anchor_label.strip().lower()
        if anchor_label:
            anchor_id = db.get_or_create_anchor(
                network_conn, anchor_label, "CAPABILITY", source="vibe"
            )
            db.link_anchor(network_conn, model_id, anchor_id, confidence=0.5)


def _extract_single_vibe(
    network_conn: sqlite3.Connection,
    model_id: str,
    raw: dict,
    extractor: object,
    build_vibe_prompt: object,
) -> object:
    """Build prompt from pre-extracted data and run vibe extraction."""
    capabilities = _get_model_capabilities(network_conn, model_id)
    family = _get_model_family(network_conn, model_id)
    param_count = _get_param_count(network_conn, model_id)

    prompt = build_vibe_prompt(  # type: ignore[operator]
        model_id=raw.get("model_id", model_id),
        author=raw.get("author", ""),
        pipeline_tag=raw.get("pipeline_tag", ""),
        tags=raw.get("tags", []),
        param_count=param_count,
        family=family,
        capabilities=capabilities,
    )
    return extractor.extract(prompt)  # type: ignore[union-attr]


def phase_c(
    ingest_conn: sqlite3.Connection,
    network_conn: sqlite3.Connection,
    vibe_min_likes: int = INGEST_VIBE_MIN_LIKES,
) -> int:
    """Run Outlines-based vibe extraction on eligible models."""
    from .extraction.vibes import VibeExtractor, build_vibe_prompt

    cursor = ingest_conn.execute(
        """SELECT model_id, raw_json FROM ingest_models
           WHERE phase_b_done = 1 AND phase_c_done = 0
             AND phase_c_attempts < ? AND likes >= ?
           ORDER BY likes DESC""",
        (VIBE_MAX_RETRIES, vibe_min_likes),
    )

    extractor = VibeExtractor()
    extractor.load()

    count = 0
    for row in cursor:
        if _shutdown:
            break

        model_id = row["model_id"]
        try:
            raw = json.loads(row["raw_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        ingest_conn.execute(
            "UPDATE ingest_models SET phase_c_attempts = phase_c_attempts + 1 WHERE model_id = ?",
            (model_id,),
        )

        try:
            result = _extract_single_vibe(
                network_conn, model_id, raw, extractor, build_vibe_prompt,
            )
            _store_vibe_result(network_conn, model_id, result)

            ingest_conn.execute(
                """UPDATE ingest_models
                   SET phase_c_done = 1, vibed_at = ?
                   WHERE model_id = ?""",
                (_now_iso(), model_id),
            )
            count += 1

            if count % 10 == 0:
                network_conn.commit()
                ingest_conn.commit()
                logger.info("Phase C: %d models vibed...", count)

        except Exception:
            logger.warning("Phase C: failed %s", model_id, exc_info=True)

    network_conn.commit()
    ingest_conn.commit()
    logger.info("Phase C: complete — %d models vibed", count)
    return count


def _get_model_capabilities(conn: sqlite3.Connection, model_id: str) -> list[str]:
    """Get capability anchors for a model from the network DB."""
    rows = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ? AND a.bank = 'CAPABILITY'""",
        (model_id,),
    ).fetchall()
    return [r[0] for r in rows]


def _get_model_family(conn: sqlite3.Connection, model_id: str) -> str:
    """Get family anchor for a model."""
    row = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ? AND a.bank = 'LINEAGE'
             AND a.category = 'family'
           LIMIT 1""",
        (model_id,),
    ).fetchone()
    return row[0] if row else "unknown"


def _get_param_count(conn: sqlite3.Connection, model_id: str) -> str:
    """Get parameter count string from metadata."""
    row = conn.execute(
        """SELECT value FROM model_metadata
           WHERE model_id = ? AND key = 'parameter_count_b'""",
        (model_id,),
    ).fetchone()
    if row:
        return f"{row[0]}B parameters"
    return "unknown"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
            phase_c(ingest_conn, network_conn, vibe_min_likes=vibe_min_likes)

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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ModelAtlas background ingestion daemon",
    )
    parser.add_argument(
        "--phase",
        default="ab",
        help="Phases to run: a, b, c, ab, abc (default: ab)",
    )
    parser.add_argument(
        "--min-likes",
        type=int,
        default=INGEST_MIN_LIKES,
        help=f"Minimum likes for Phase A (default: {INGEST_MIN_LIKES})",
    )
    parser.add_argument(
        "--vibe-min-likes",
        type=int,
        default=INGEST_VIBE_MIN_LIKES,
        help=f"Minimum likes for Phase C vibes (default: {INGEST_VIBE_MIN_LIKES})",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode (loop with 24h sleep)",
    )
    parser.add_argument(
        "--daemon-sleep",
        type=int,
        default=86400,
        help="Sleep seconds between daemon cycles (default: 86400)",
    )
    parser.add_argument(
        "--seed",
        nargs="*",
        metavar="PASS",
        help="Seed the network via multi-pass HF streaming. Passes: core, expand, niche (default: all)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show ingest progress and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.status:
        ingest_conn = db_ingest.get_connection()
        db_ingest.init_db(ingest_conn)
        print_status(ingest_conn)
        ingest_conn.close()
        return

    if args.seed is not None:
        from .ingest_seed import seed

        network_conn = db.get_connection()
        db.init_db(network_conn)
        pass_names = args.seed if args.seed else None  # empty list = all
        results = seed(network_conn, passes=pass_names)
        stats = db.network_stats(network_conn)
        network_conn.close()
        print("\nSeed complete:")
        for name, count in results.items():
            print(f"  {name}: {count} models indexed")
        print(
            f"\nNetwork total: {stats['total_models']} models, {stats['total_anchors']} anchors"
        )
        return

    run(
        phases=args.phase,
        min_likes=args.min_likes,
        vibe_min_likes=args.vibe_min_likes,
        daemon=args.daemon,
        daemon_sleep=args.daemon_sleep,
    )


if __name__ == "__main__":
    main()
