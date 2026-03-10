"""Tests for the multi-pass network seeder."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from model_atlas.ingest_seed import (
    SEED_PASSES,
    _hf_model_to_input,
    _make_tz_aware,
    _passes_date_filter,
    _passes_seed_filters,
    _safetensors_to_dict,
)

# ---------------------------------------------------------------------------
# _make_tz_aware
# ---------------------------------------------------------------------------


class TestMakeTzAware:
    def test_naive_becomes_utc(self):
        naive = datetime(2025, 1, 15, 12, 0, 0)
        result = _make_tz_aware(naive)
        assert result.tzinfo == timezone.utc
        assert result.year == 2025

    def test_aware_unchanged(self):
        aware = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = _make_tz_aware(aware)
        assert result is aware


# ---------------------------------------------------------------------------
# _passes_date_filter
# ---------------------------------------------------------------------------


class TestPassesDateFilter:
    def test_no_min_always_passes(self):
        assert _passes_date_filter(datetime(2020, 1, 1), None) is True

    def test_no_created_at_passes(self):
        min_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert _passes_date_filter(None, min_dt) is True

    def test_non_datetime_passes(self):
        """Non-datetime objects (e.g. strings) pass through."""
        min_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert _passes_date_filter("2024-01-01", min_dt) is True

    def test_after_min_passes(self):
        created = datetime(2024, 6, 1, tzinfo=timezone.utc)
        min_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert _passes_date_filter(created, min_dt) is True

    def test_before_min_fails(self):
        created = datetime(2022, 6, 1, tzinfo=timezone.utc)
        min_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert _passes_date_filter(created, min_dt) is False

    def test_exact_min_passes(self):
        dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert _passes_date_filter(dt, dt) is True

    def test_naive_created_vs_aware_min(self):
        """Naive datetimes get UTC assumed, then compared."""
        created = datetime(2024, 1, 1)  # naive
        min_dt = datetime(2023, 6, 1, tzinfo=timezone.utc)
        assert _passes_date_filter(created, min_dt) is True


# ---------------------------------------------------------------------------
# _safetensors_to_dict
# ---------------------------------------------------------------------------


class TestSafetensorsToDict:
    def test_none_returns_none(self):
        assert _safetensors_to_dict(None) is None

    def test_dict_passthrough(self):
        d = {"parameters": {"total": 7_000_000_000}}
        assert _safetensors_to_dict(d) is d

    def test_object_with_parameters(self):
        obj = SimpleNamespace(parameters={"total": 7_000_000_000})
        result = _safetensors_to_dict(obj)
        assert result == {"parameters": {"total": 7_000_000_000}}

    def test_object_with_total(self):
        obj = SimpleNamespace(total=7_000_000_000)
        result = _safetensors_to_dict(obj)
        assert result == {"total": 7_000_000_000}

    def test_object_with_both(self):
        obj = SimpleNamespace(parameters={"layer": 32}, total=7_000_000_000)
        result = _safetensors_to_dict(obj)
        assert result["parameters"] == {"layer": 32}
        assert result["total"] == 7_000_000_000

    def test_empty_object_returns_none(self):
        """Object with no relevant attrs returns None."""
        obj = SimpleNamespace(unrelated="data")
        assert _safetensors_to_dict(obj) is None


# ---------------------------------------------------------------------------
# _hf_model_to_input
# ---------------------------------------------------------------------------


def _make_hf_model(**kwargs):
    """Create a mock HF model listing object."""
    defaults = {
        "id": "test/Model-7B",
        "author": "test",
        "pipeline_tag": "text-generation",
        "tags": ["text-generation", "pytorch"],
        "library_name": "transformers",
        "likes": 500,
        "downloads": 10000,
        "created_at": "2025-01-15T00:00:00Z",
        "safetensors": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestHfModelToInput:
    def test_basic_conversion(self):
        model = _make_hf_model()
        inp = _hf_model_to_input(model)
        assert inp.model_id == "test/Model-7B"
        assert inp.author == "test"
        assert inp.pipeline_tag == "text-generation"
        assert inp.tags == ["text-generation", "pytorch"]
        assert inp.library_name == "transformers"
        assert inp.likes == 500
        assert inp.downloads == 10000
        assert inp.source == "huggingface"
        assert inp.config is None

    def test_none_fields_become_empty(self):
        model = _make_hf_model(
            author=None,
            pipeline_tag=None,
            tags=None,
            library_name=None,
            likes=None,
            downloads=None,
            created_at=None,
        )
        inp = _hf_model_to_input(model)
        assert inp.author == ""
        assert inp.pipeline_tag == ""
        assert inp.tags == []
        assert inp.library_name == ""
        assert inp.likes == 0
        assert inp.downloads == 0
        assert inp.created_at is None

    def test_safetensors_dict_passed_through(self):
        st = {"parameters": {"total": 7_000_000_000}}
        model = _make_hf_model(safetensors=st)
        inp = _hf_model_to_input(model)
        assert inp.safetensors_info == st

    def test_license_from_attr(self):
        model = _make_hf_model()
        model.license = "mit"
        inp = _hf_model_to_input(model)
        assert inp.license_str == "mit"

    def test_no_license_attr(self):
        model = _make_hf_model()
        # SimpleNamespace doesn't have license by default
        inp = _hf_model_to_input(model)
        assert inp.license_str == ""

    def test_config_extracted_from_model(self):
        """Config dict from HF model object is passed through."""
        config = {"architectures": ["BertModel"], "model_type": "bert"}
        model = _make_hf_model(config=config)
        inp = _hf_model_to_input(model)
        assert inp.config == config
        assert inp.config["architectures"] == ["BertModel"]

    def test_config_none_when_absent(self):
        """Models without config attr get config=None."""
        model = _make_hf_model()
        # SimpleNamespace won't have 'config' unless we add it
        if hasattr(model, "config"):
            delattr(model, "config")
        inp = _hf_model_to_input(model)
        assert inp.config is None


# ---------------------------------------------------------------------------
# _passes_seed_filters
# ---------------------------------------------------------------------------


class TestPassesSeedFilters:
    def test_passes_all_filters(self):
        model = _make_hf_model(likes=200, downloads=500)
        result = _passes_seed_filters(model, 100, 100, None, set())
        assert result == "test/Model-7B"

    def test_below_min_likes_returns_none(self):
        model = _make_hf_model(likes=5)
        assert _passes_seed_filters(model, 100, 0, None, set()) is None

    def test_already_existing_returns_none(self):
        model = _make_hf_model(likes=200)
        existing = {"test/Model-7B"}
        assert _passes_seed_filters(model, 100, 0, None, existing) is None

    def test_below_min_downloads_returns_none(self):
        model = _make_hf_model(likes=200, downloads=10)
        assert _passes_seed_filters(model, 100, 500, None, set()) is None

    def test_fails_date_filter_returns_none(self):
        model = _make_hf_model(likes=200, downloads=500)
        model.created_at = datetime(2022, 1, 1, tzinfo=timezone.utc)
        min_created = datetime(2023, 6, 1, tzinfo=timezone.utc)
        assert _passes_seed_filters(model, 100, 100, min_created, set()) is None

    def test_empty_model_id_returns_none(self):
        model = _make_hf_model(id="")
        assert _passes_seed_filters(model, 0, 0, None, set()) is None

    def test_none_likes_treated_as_zero(self):
        model = _make_hf_model(likes=None)
        assert _passes_seed_filters(model, 1, 0, None, set()) is None


# ---------------------------------------------------------------------------
# SEED_PASSES structure
# ---------------------------------------------------------------------------


class TestSeedPassesConfig:
    def test_has_three_passes(self):
        assert len(SEED_PASSES) == 3

    def test_pass_names(self):
        names = [p["name"] for p in SEED_PASSES]
        assert names == ["core", "expand", "niche"]

    def test_likes_descending(self):
        """Each pass has progressively lower min_likes threshold."""
        likes = [p["min_likes"] for p in SEED_PASSES]
        assert likes[0] > likes[1] > likes[2]

    def test_core_has_no_tag_filter(self):
        core = SEED_PASSES[0]
        assert core["pipeline_tags"] is None

    def test_niche_has_tag_filter(self):
        niche = SEED_PASSES[2]
        assert niche["pipeline_tags"] is not None
        assert "text-generation" in niche["pipeline_tags"]

    def test_all_passes_have_required_keys(self):
        required = {
            "name",
            "description",
            "min_likes",
            "min_downloads",
            "min_created",
            "pipeline_tags",
        }
        for p in SEED_PASSES:
            assert required.issubset(p.keys()), f"Pass '{p['name']}' missing keys"
