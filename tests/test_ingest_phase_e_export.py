"""Tests for Phase E export (priority ordering, query generation, sharding)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from model_atlas import db
from model_atlas.ingest_phase_e_export import (
    _build_search_queries,
    _build_one_record,
    export_phase_e,
    get_priority_models,
)


@pytest.fixture
def network_conn():
    """In-memory network database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    db.init_db(conn)
    return conn


def _add_model(conn, model_id, author="test", downloads=0, likes=0):
    db.insert_model(conn, model_id, author=author, source="huggingface")
    if downloads:
        db.set_metadata(conn, model_id, "downloads", str(downloads), "int")
    if likes:
        db.set_metadata(conn, model_id, "likes", str(likes), "int")


# ---------------------------------------------------------------------------
# _build_search_queries
# ---------------------------------------------------------------------------


class TestBuildSearchQueries:
    def test_basic_queries(self):
        queries = _build_search_queries(
            "meta-llama/Llama-3.1-8B", "meta-llama", "text-generation",
            "Llama-family", "8.0B", [],
        )
        assert len(queries) >= 2
        assert any("Llama-3.1-8B" in q for q in queries)
        assert any("benchmark" in q or "evaluation" in q for q in queries)

    def test_domain_anchor_generates_third_query(self):
        queries = _build_search_queries(
            "test/model", "test", "text-generation",
            "", "", ["code-domain"],
        )
        assert len(queries) == 3
        assert any("code" in q for q in queries)

    def test_family_fallback_third_query(self):
        queries = _build_search_queries(
            "test/model", "test", "", "Mistral-family", "", [],
        )
        assert len(queries) >= 2
        assert any("Mistral" in q for q in queries)

    def test_max_three_queries(self):
        queries = _build_search_queries(
            "a/b", "a", "text-generation", "Llama-family", "7B",
            ["code-domain", "medical-domain"],
        )
        assert len(queries) <= 3

    def test_no_author_still_works(self):
        queries = _build_search_queries(
            "standalone-model", "", "text-generation", "", "", [],
        )
        assert len(queries) >= 2
        assert all(q.strip() for q in queries)


# ---------------------------------------------------------------------------
# get_priority_models
# ---------------------------------------------------------------------------


class TestGetPriorityModels:
    def test_high_downloads_first(self, network_conn):
        _add_model(network_conn, "low/model", downloads=50)
        _add_model(network_conn, "high/model", downloads=500)
        _add_model(network_conn, "mid/model", downloads=200)

        ids = get_priority_models(network_conn, min_downloads=100)
        assert ids == ["high/model", "mid/model"]

    def test_full_corpus_includes_low_downloads(self, network_conn):
        _add_model(network_conn, "low/model", downloads=10, likes=5)
        _add_model(network_conn, "high/model", downloads=500)

        ids = get_priority_models(network_conn, min_downloads=100, full_corpus=True)
        assert ids[0] == "high/model"
        assert "low/model" in ids

    def test_skips_already_enriched(self, network_conn):
        _add_model(network_conn, "done/model", downloads=500)
        db.set_metadata(network_conn, "done/model", "web_enriched", "true", "str")
        _add_model(network_conn, "new/model", downloads=500)

        ids = get_priority_models(network_conn, min_downloads=100, skip_existing=True)
        assert "done/model" not in ids
        assert "new/model" in ids

    def test_empty_db(self, network_conn):
        ids = get_priority_models(network_conn, min_downloads=100)
        assert ids == []


# ---------------------------------------------------------------------------
# _build_one_record
# ---------------------------------------------------------------------------


class TestBuildOneRecord:
    def test_record_structure(self, network_conn):
        _add_model(network_conn, "test/model", author="testorg")
        db.set_metadata(network_conn, "test/model", "pipeline_tag", "text-generation", "str")
        bank_vocab = {"CAPABILITY": ["chat", "reasoning"], "QUALITY": ["trending"]}

        record = _build_one_record(network_conn, "test/model", bank_vocab)
        assert record["model_id"] == "test/model"
        assert len(record["search_queries"]) >= 2
        assert record["banks"] == bank_vocab
        assert record["existing_metadata"]["author"] == "testorg"
        assert record["existing_metadata"]["pipeline_tag"] == "text-generation"
        assert isinstance(record["existing_metadata"]["current_anchors"], list)

    def test_record_includes_current_anchors(self, network_conn):
        _add_model(network_conn, "test/model")
        aid = db.get_or_create_anchor(network_conn, "chat", "CAPABILITY", source="bootstrap")
        db.link_anchor(network_conn, "test/model", aid)

        record = _build_one_record(network_conn, "test/model", {})
        assert "chat" in record["existing_metadata"]["current_anchors"]


# ---------------------------------------------------------------------------
# export_phase_e (integration)
# ---------------------------------------------------------------------------


class TestExportPhaseE:
    def test_sharded_round_robin(self, network_conn, tmp_path):
        for i in range(5):
            _add_model(network_conn, f"test/model-{i}", downloads=500)

        n = export_phase_e(network_conn, num_shards=2, min_downloads=100, work_dir=tmp_path)
        assert n == 5
        shard_0 = (tmp_path / "shard_0.jsonl").read_text().strip().split("\n")
        shard_1 = (tmp_path / "shard_1.jsonl").read_text().strip().split("\n")
        assert len(shard_0) == 3
        assert len(shard_1) == 2

    def test_bank_filter(self, network_conn, tmp_path):
        _add_model(network_conn, "test/model", downloads=500)
        export_phase_e(
            network_conn, num_shards=1, banks=["CAPABILITY", "QUALITY"],
            min_downloads=100, work_dir=tmp_path,
        )
        record = json.loads((tmp_path / "shard_0.jsonl").read_text().strip())
        # Only requested banks should appear (if they have anchors in DB)
        for bank in record["banks"]:
            assert bank in ("CAPABILITY", "QUALITY")

    def test_empty_returns_zero(self, network_conn, tmp_path):
        n = export_phase_e(network_conn, num_shards=1, min_downloads=100, work_dir=tmp_path)
        assert n == 0

    def test_records_are_valid_json(self, network_conn, tmp_path):
        _add_model(network_conn, "test/model", downloads=500)
        export_phase_e(network_conn, num_shards=1, min_downloads=100, work_dir=tmp_path)
        for line in (tmp_path / "shard_0.jsonl").read_text().strip().split("\n"):
            record = json.loads(line)
            assert "model_id" in record
            assert "search_queries" in record
            assert "banks" in record
            assert "existing_metadata" in record
