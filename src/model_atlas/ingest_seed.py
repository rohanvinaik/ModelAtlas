"""Multi-pass network seeder for ModelAtlas.

Streams HuggingFace API sorted by likes (descending), applies per-pass
client-side filters, runs Tier 1+2 extraction, and commits after each pass.
The network is queryable after each pass completes.

Usage:
  python -m model_atlas.ingest --seed               # all passes
  python -m model_atlas.ingest --seed core expand    # specific passes
"""

from __future__ import annotations

import logging
import signal
import sqlite3
from datetime import datetime, timezone

from . import db
from .config import INGEST_BATCH_SIZE
from .extraction.deterministic import ModelInput
from .extraction.pipeline import extract_and_store

logger = logging.getLogger(__name__)

# Graceful shutdown flag (shared with ingest.py via import)
_shutdown = False


def _handle_signal(signum: int, frame: object) -> None:
    global _shutdown
    logger.info("Received signal %d, finishing current batch...", signum)
    _shutdown = True


# ---------------------------------------------------------------------------
# Seed pass definitions
# ---------------------------------------------------------------------------

# Passes are processed in order. Each pass streams the HF API sorted by likes
# (always descending) and applies client-side filters. Models already in the
# network DB are skipped. The network is usable after each pass commits.
SEED_PASSES = [
    {
        "name": "core",
        "description": "Top models everyone has heard of",
        "min_likes": 100,
        "min_downloads": 0,
        "min_created": None,
        "pipeline_tags": None,
    },
    {
        "name": "expand",
        "description": "Broader coverage, recent models with traction",
        "min_likes": 10,
        "min_downloads": 100,
        "min_created": "2023-01-01",
        "pipeline_tags": None,
    },
    {
        "name": "niche",
        "description": "Niche fine-tunes in key categories",
        "min_likes": 3,
        "min_downloads": 50,
        "min_created": "2023-06-01",
        "pipeline_tags": [
            "text-generation",
            "text2text-generation",
            "image-text-to-text",
            "feature-extraction",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tz_aware(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (defaults to UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _passes_date_filter(model_created_at: object, min_created: datetime | None) -> bool:
    """Check if a model's creation date passes the minimum date filter."""
    if not min_created or not model_created_at:
        return True
    created = model_created_at if isinstance(model_created_at, datetime) else None
    if created is None:
        return True
    return _make_tz_aware(created) >= _make_tz_aware(min_created)


def _safetensors_to_dict(info: object) -> dict | None:
    """Convert safetensors info object to a serializable dict."""
    if info is None:
        return None
    if isinstance(info, dict):
        return info
    result = {}
    if hasattr(info, "parameters"):
        result["parameters"] = info.parameters
    if hasattr(info, "total"):
        result["total"] = info.total
    return result if result else None


def _hf_model_to_input(model: object) -> ModelInput:
    """Convert an HF model listing object to a ModelInput for extraction."""
    safetensors_info = None
    if hasattr(model, "safetensors") and model.safetensors:
        safetensors_info = _safetensors_to_dict(model.safetensors)

    return ModelInput(
        model_id=model.id or "",  # type: ignore[union-attr]
        author=model.author or "",  # type: ignore[union-attr]
        pipeline_tag=model.pipeline_tag or "",  # type: ignore[union-attr]
        tags=list(model.tags or []),  # type: ignore[union-attr]
        library_name=model.library_name or "",  # type: ignore[union-attr]
        likes=model.likes or 0,  # type: ignore[union-attr]
        downloads=model.downloads or 0,  # type: ignore[union-attr]
        created_at=str(model.created_at) if model.created_at else None,  # type: ignore[union-attr]
        license_str=getattr(model, "license", "") or "",
        safetensors_info=safetensors_info,
        config=getattr(model, "config", None),
        source="huggingface",
    )


# ---------------------------------------------------------------------------
# Core seed logic
# ---------------------------------------------------------------------------


def _open_hf_streams(pipeline_tags: list[str] | None) -> list:
    """Open one or more HF API model streams sorted by likes."""
    from huggingface_hub import HfApi

    api = HfApi()
    if pipeline_tags:
        return [
            api.list_models(
                full=True, sort="likes", pipeline_tag=tag, fetch_config=True
            )
            for tag in pipeline_tags
        ]
    return [api.list_models(full=True, sort="likes", fetch_config=True)]


def _passes_seed_filters(
    model: object,
    min_likes: int,
    min_downloads: int,
    min_created: datetime | None,
    existing: set[str],
) -> str | None:
    """Check if a model passes seed filters. Returns model_id or None."""
    likes = getattr(model, "likes", 0) or 0
    if likes < min_likes:
        return None  # sorted by likes desc — everything after is below threshold

    model_id = getattr(model, "id", "") or ""
    if not model_id or model_id in existing:
        return None

    if (getattr(model, "downloads", 0) or 0) < min_downloads:
        return None

    if not _passes_date_filter(getattr(model, "created_at", None), min_created):
        return None

    return model_id


def _seed_single_pass(
    network_conn: sqlite3.Connection,
    pass_def: dict,
    existing: set[str],
) -> int:
    """Execute a single seed pass. Returns count of models indexed."""
    pass_name = pass_def["name"]
    min_likes = pass_def["min_likes"]
    min_downloads = pass_def["min_downloads"]
    min_created_str = pass_def["min_created"]
    min_created = datetime.fromisoformat(min_created_str) if min_created_str else None
    streams = _open_hf_streams(pass_def["pipeline_tags"])

    count = 0
    for stream in streams:
        if _shutdown:
            break
        for model in stream:
            if _shutdown:
                break

            model_id = _passes_seed_filters(
                model,
                min_likes,
                min_downloads,
                min_created,
                existing,
            )
            if model_id is None:
                # If below min_likes, stop stream (sorted desc)
                if (getattr(model, "likes", 0) or 0) < min_likes:
                    break
                continue

            try:
                inp = _hf_model_to_input(model)
                extract_and_store(network_conn, inp)
                existing.add(model_id)
                count += 1

                if count % INGEST_BATCH_SIZE == 0:
                    network_conn.commit()
                    logger.info(
                        "Seed pass '%s': %d models indexed...", pass_name, count
                    )
            except Exception:
                logger.warning("Seed: failed %s", model_id, exc_info=True)

    return count


def seed(
    network_conn: sqlite3.Connection,
    passes: list[str] | None = None,
) -> dict[str, int]:
    """Multi-pass network seeder. Streams HF API and extracts directly.

    Each pass fetches models sorted by likes (descending), applies client-side
    filters, runs Tier 1+2 extraction, and commits. The network is queryable
    after each pass.
    """
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    existing = {
        row[0] for row in network_conn.execute("SELECT model_id FROM models").fetchall()
    }
    logger.info("Seed: %d models already in network", len(existing))

    results: dict[str, int] = {}
    for pass_def in SEED_PASSES:
        pass_name = str(pass_def["name"])
        if passes and pass_name not in passes:
            continue
        if _shutdown:
            break

        logger.info(
            "Seed pass '%s': %s (min_likes=%d)",
            pass_name,
            pass_def["description"],
            pass_def["min_likes"],
        )

        count: int = _seed_single_pass(network_conn, pass_def, existing)
        network_conn.commit()
        results[pass_name] = count

        stats = db.network_stats(network_conn)
        logger.info(
            "Seed pass '%s' complete: %d new models (network total: %d)",
            pass_name,
            count,
            stats["total_models"],
        )

    return results
