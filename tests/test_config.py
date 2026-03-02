"""Tests for configuration constants."""

from __future__ import annotations

from pathlib import Path

from model_atlas.config import (
    ANCHOR_SLICE,
    CACHE_DIR,
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_INDEX_SIZE,
    DEFAULT_RESULT_LIMIT,
    INGEST_BATCH_SIZE,
    INGEST_DB_PATH,
    INGEST_MIN_LIKES,
    INGEST_VIBE_MIN_LIKES,
    LINK_WEIGHTS,
    MAX_CARD_TEXT_LENGTH,
    MODEL_CARD_CACHE_DIR,
    MODEL_CARD_TTL,
    NEIGHBOR_SLICE,
    NETWORK_DB_PATH,
    SPREAD_DECAY,
    SPREAD_MAX_DEPTH,
    VIBE_MAX_RETRIES,
    VIBE_MODEL_NAME,
    WEIGHT_ANCHOR,
    WEIGHT_BANK,
    WEIGHT_FUZZY,
    WEIGHT_SPREAD,
)


class TestPathConstants:
    def test_cache_dir_is_path(self):
        assert isinstance(CACHE_DIR, Path)

    def test_model_card_cache_dir_is_path(self):
        assert isinstance(MODEL_CARD_CACHE_DIR, Path)

    def test_network_db_path_is_path(self):
        assert isinstance(NETWORK_DB_PATH, Path)

    def test_ingest_db_path_is_path(self):
        assert isinstance(INGEST_DB_PATH, Path)

    def test_cache_dir_under_home(self):
        """Cache directory should be under the user's home directory."""
        assert CACHE_DIR.parts[0] == "/"
        assert ".cache" in CACHE_DIR.parts
        assert "model-atlas" in CACHE_DIR.parts

    def test_model_card_cache_is_under_cache_dir(self):
        assert MODEL_CARD_CACHE_DIR.parent == CACHE_DIR

    def test_network_db_is_under_cache_dir(self):
        assert NETWORK_DB_PATH.parent == CACHE_DIR

    def test_ingest_db_is_under_cache_dir(self):
        assert INGEST_DB_PATH.parent == CACHE_DIR


class TestTTLConstants:
    def test_model_card_ttl_positive(self):
        assert MODEL_CARD_TTL > 0

    def test_model_card_ttl_is_24_hours(self):
        assert MODEL_CARD_TTL == 86400


class TestLimitConstants:
    def test_default_candidate_limit_reasonable(self):
        assert 100 <= DEFAULT_CANDIDATE_LIMIT <= 10000

    def test_default_result_limit_reasonable(self):
        assert 5 <= DEFAULT_RESULT_LIMIT <= 100

    def test_result_limit_less_than_candidate_limit(self):
        assert DEFAULT_RESULT_LIMIT < DEFAULT_CANDIDATE_LIMIT

    def test_max_card_text_length_positive(self):
        assert MAX_CARD_TEXT_LENGTH > 0

    def test_default_index_size_positive(self):
        assert DEFAULT_INDEX_SIZE > 0


class TestScoringWeights:
    def test_weights_sum_to_one(self):
        total = WEIGHT_BANK + WEIGHT_ANCHOR + WEIGHT_SPREAD + WEIGHT_FUZZY
        assert abs(total - 1.0) < 1e-9

    def test_all_weights_positive(self):
        assert WEIGHT_BANK > 0
        assert WEIGHT_ANCHOR > 0
        assert WEIGHT_SPREAD > 0
        assert WEIGHT_FUZZY > 0

    def test_all_weights_less_than_one(self):
        assert WEIGHT_BANK < 1.0
        assert WEIGHT_ANCHOR < 1.0
        assert WEIGHT_SPREAD < 1.0
        assert WEIGHT_FUZZY < 1.0


class TestSpreadingConstants:
    def test_decay_between_zero_and_one(self):
        assert 0 < SPREAD_DECAY < 1.0

    def test_max_depth_positive(self):
        assert SPREAD_MAX_DEPTH > 0

    def test_neighbor_slice_positive(self):
        assert NEIGHBOR_SLICE > 0

    def test_anchor_slice_positive(self):
        assert ANCHOR_SLICE > 0


class TestLinkWeights:
    def test_link_weights_is_dict(self):
        assert isinstance(LINK_WEIGHTS, dict)

    def test_all_link_weights_between_zero_and_one(self):
        for relation, weight in LINK_WEIGHTS.items():
            assert 0 < weight <= 1.0, f"{relation} weight {weight} out of range"

    def test_expected_relations_present(self):
        assert "fine_tuned_from" in LINK_WEIGHTS
        assert "quantized_from" in LINK_WEIGHTS
        assert "variant_of" in LINK_WEIGHTS
        assert "same_family" in LINK_WEIGHTS
        assert "predecessor" in LINK_WEIGHTS
        assert "successor" in LINK_WEIGHTS

    def test_fine_tuned_from_highest_weight(self):
        """Fine-tune relationship should be the strongest link."""
        assert LINK_WEIGHTS["fine_tuned_from"] >= max(
            v for k, v in LINK_WEIGHTS.items() if k != "fine_tuned_from"
        )


class TestIngestConstants:
    def test_ingest_batch_size_positive(self):
        assert INGEST_BATCH_SIZE > 0

    def test_ingest_min_likes_non_negative(self):
        assert INGEST_MIN_LIKES >= 0

    def test_vibe_min_likes_greater_than_ingest_min(self):
        assert INGEST_VIBE_MIN_LIKES >= INGEST_MIN_LIKES

    def test_vibe_model_name_is_string(self):
        assert isinstance(VIBE_MODEL_NAME, str)
        assert len(VIBE_MODEL_NAME) > 0

    def test_vibe_max_retries_positive(self):
        assert VIBE_MAX_RETRIES > 0
