"""Disk-based caching for model cards and metadata."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import MODEL_CARD_CACHE_DIR, MODEL_CARD_TTL


def _safe_model_path(model_id: str) -> Path:
    """Convert model_id (e.g. 'meta-llama/Llama-3') to a safe filesystem path."""
    return MODEL_CARD_CACHE_DIR / model_id.replace("/", "__")


def get_cached_model(model_id: str) -> dict | None:
    """Return cached model data if it exists and is fresh, else None."""
    path = _safe_model_path(model_id) / "meta.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data.get("_cached_at", 0) > MODEL_CARD_TTL:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def store_model(model_id: str, data: dict) -> None:
    """Cache model metadata to disk."""
    dir_path = _safe_model_path(model_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    data["_cached_at"] = time.time()
    (dir_path / "meta.json").write_text(json.dumps(data, default=str))


def get_cached_card_text(model_id: str) -> str | None:
    """Return cached model card text if available."""
    path = _safe_model_path(model_id) / "card.txt"
    if not path.exists():
        return None
    try:
        text = path.read_text()
        # Check if accompanying meta is still fresh
        meta = get_cached_model(model_id)
        if meta is None:
            return None
        return text
    except OSError:
        return None


def store_card_text(model_id: str, text: str) -> None:
    """Cache model card text to disk."""
    dir_path = _safe_model_path(model_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "card.txt").write_text(text)


def clear_cache() -> int:
    """Remove all cached data. Returns number of entries removed."""
    if not MODEL_CARD_CACHE_DIR.exists():
        return 0
    count = 0
    for entry in MODEL_CARD_CACHE_DIR.iterdir():
        if entry.is_dir():
            for f in entry.iterdir():
                f.unlink()
            entry.rmdir()
            count += 1
    return count
