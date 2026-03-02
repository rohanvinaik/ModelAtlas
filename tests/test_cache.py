"""Tests for disk-based model card caching."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

from model_atlas.cache import (
    _safe_model_path,
    clear_cache,
    get_cached_card_text,
    get_cached_model,
    store_card_text,
    store_model,
)


class TestSafeModelPath:
    def test_slash_replaced_with_double_underscore(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            path = _safe_model_path("meta-llama/Llama-3.1-8B")
            assert "__" in path.name
            assert "/" not in path.name

    def test_no_slash_model_id(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            path = _safe_model_path("simple-model")
            assert path.name == "simple-model"

    def test_result_is_under_cache_dir(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            path = _safe_model_path("org/model")
            assert path.parent == tmp_path


class TestStoreAndRetrieveModel:
    def test_store_then_get(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            data = {"model_id": "test/Model", "likes": 42}
            store_model("test/Model", data)
            result = get_cached_model("test/Model")
            assert result is not None
            assert result["model_id"] == "test/Model"
            assert result["likes"] == 42

    def test_cached_at_timestamp_added(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            data = {"model_id": "test/Model"}
            store_model("test/Model", data)
            result = get_cached_model("test/Model")
            assert result is not None
            assert "_cached_at" in result
            assert isinstance(result["_cached_at"], float)

    def test_get_nonexistent_returns_none(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            result = get_cached_model("nonexistent/Model")
            assert result is None

    def test_ttl_expired_returns_none(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            data = {"model_id": "test/Model"}
            store_model("test/Model", data)
            # Manually backdate the _cached_at to simulate expiry
            meta_path = _safe_model_path("test/Model") / "meta.json"
            stored = json.loads(meta_path.read_text())
            stored["_cached_at"] = time.time() - 200_000  # well past 24h TTL
            meta_path.write_text(json.dumps(stored))
            result = get_cached_model("test/Model")
            assert result is None

    def test_corrupt_json_returns_none(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            dir_path = _safe_model_path("test/Model")
            dir_path.mkdir(parents=True)
            (dir_path / "meta.json").write_text("{invalid json!!!")
            result = get_cached_model("test/Model")
            assert result is None

    def test_overwrite_existing_cache(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            store_model("test/Model", {"version": 1})
            store_model("test/Model", {"version": 2})
            result = get_cached_model("test/Model")
            assert result is not None
            assert result["version"] == 2


class TestCardTextCache:
    def test_store_and_retrieve_card_text(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            # Card text requires a fresh meta.json to be considered valid
            store_model("test/Model", {"model_id": "test/Model"})
            store_card_text("test/Model", "This is a model card.")
            result = get_cached_card_text("test/Model")
            assert result == "This is a model card."

    def test_card_text_without_meta_returns_none(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            # Store card text but no model metadata
            dir_path = _safe_model_path("test/Model")
            dir_path.mkdir(parents=True)
            (dir_path / "card.txt").write_text("orphan card text")
            result = get_cached_card_text("test/Model")
            assert result is None

    def test_card_text_with_expired_meta_returns_none(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            store_model("test/Model", {"model_id": "test/Model"})
            store_card_text("test/Model", "Some text")
            # Expire the meta
            meta_path = _safe_model_path("test/Model") / "meta.json"
            stored = json.loads(meta_path.read_text())
            stored["_cached_at"] = time.time() - 200_000
            meta_path.write_text(json.dumps(stored))
            result = get_cached_card_text("test/Model")
            assert result is None

    def test_nonexistent_card_returns_none(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            result = get_cached_card_text("nonexistent/Model")
            assert result is None


class TestClearCache:
    def test_clear_empty_cache(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            count = clear_cache()
            assert count == 0

    def test_clear_nonexistent_dir(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", nonexistent):
            count = clear_cache()
            assert count == 0

    def test_clear_populated_cache(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            store_model("org/model-a", {"id": "a"})
            store_model("org/model-b", {"id": "b"})
            store_card_text("org/model-a", "card a")
            count = clear_cache()
            assert count == 2
            # Verify data is gone
            assert get_cached_model("org/model-a") is None
            assert get_cached_model("org/model-b") is None

    def test_clear_returns_count_of_model_dirs(self, tmp_path):
        with patch("model_atlas.cache.MODEL_CARD_CACHE_DIR", tmp_path):
            store_model("a/m1", {"id": "1"})
            store_model("a/m2", {"id": "2"})
            store_model("a/m3", {"id": "3"})
            count = clear_cache()
            assert count == 3
