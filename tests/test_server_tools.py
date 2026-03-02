"""Tests for MCP server tools with mocked DB."""

from __future__ import annotations

import json

from model_atlas import db
from model_atlas.server import (
    _find_models_without_vibes,
    hf_compare_models,
    hf_get_model_detail,
    hf_index_status,
    set_model_vibe,
)


class _NoCloseConn:
    """Wrapper that delegates everything to conn but makes close() a no-op."""

    def __init__(self, conn):
        self._conn = conn

    def close(self):
        pass  # no-op to keep test connection alive

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _patch_db(monkeypatch, conn):
    """Patch get_connection to return a non-closing wrapper around test conn."""
    monkeypatch.setattr(
        "model_atlas.server.db.get_connection", lambda: _NoCloseConn(conn)
    )


class TestSetModelVibe:
    def test_set_vibe_stores_summary(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(
            set_model_vibe(
                "meta-llama/Llama-3.1-8B-Instruct",
                "A versatile 8B instruct model excelling at general tasks.",
            )
        )
        assert result["status"] == "updated"
        assert (
            result["vibe_summary"]
            == "A versatile 8B instruct model excelling at general tasks."
        )

        # Verify stored in metadata
        model = db.get_model(populated_conn, "meta-llama/Llama-3.1-8B-Instruct")
        vibe_meta = model["metadata"].get("vibe_summary", {})
        assert (
            vibe_meta["value"]
            == "A versatile 8B instruct model excelling at general tasks."
        )

    def test_set_vibe_with_extra_anchors(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(
            set_model_vibe(
                "meta-llama/Llama-3.1-8B-Instruct",
                "Great for tool-calling workflows.",
                extra_anchors=["tool-calling", "orchestration"],
            )
        )
        assert result["status"] == "updated"
        assert "tool-calling" in result["anchors_added"]
        assert "orchestration" in result["anchors_added"]

        # Verify anchors are linked
        anchors = db.get_anchor_set(populated_conn, "meta-llama/Llama-3.1-8B-Instruct")
        assert "tool-calling" in anchors
        assert "orchestration" in anchors

    def test_set_vibe_missing_model(self, conn, monkeypatch):
        _patch_db(monkeypatch, conn)
        result = json.loads(set_model_vibe("nonexistent/Model", "some vibe"))
        assert "error" in result

    def test_set_vibe_new_anchor_gets_vibes_source(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        set_model_vibe(
            "meta-llama/Llama-3.1-8B-Instruct",
            "Custom vibe.",
            extra_anchors=["brand-new-anchor"],
        )
        row = populated_conn.execute(
            "SELECT source FROM anchors WHERE label = 'brand-new-anchor'"
        ).fetchone()
        assert row is not None
        assert row["source"] == "vibes"


class TestHfGetModelDetail:
    def test_returns_network_data(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(hf_get_model_detail("meta-llama/Llama-3.1-8B-Instruct"))
        assert result["source"] == "network"
        assert result["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert "positions" in result
        assert "anchors" in result


class TestHfCompareModels:
    def test_compare_returns_structure(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(
            hf_compare_models(
                ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-Coder-1.5B"]
            )
        )
        assert len(result["models"]) == 2
        assert "shared_anchors" in result
        assert "jaccard_similarity" in result
        assert "bank_deltas" in result


class TestHfIndexStatus:
    def test_returns_stats(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(hf_index_status())
        assert result["total_models"] == 4
        assert result["total_anchors"] > 0


class TestFindModelsWithoutVibes:
    def test_finds_models_lacking_vibes(self, populated_conn):
        # None of the fixture models have vibes
        missing = _find_models_without_vibes(
            populated_conn,
            ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-Coder-1.5B"],
        )
        assert "meta-llama/Llama-3.1-8B-Instruct" in missing
        assert "Qwen/Qwen2.5-Coder-1.5B" in missing

    def test_excludes_models_with_vibes(self, populated_conn):
        # Add a vibe to one model
        db.set_metadata(
            populated_conn,
            "meta-llama/Llama-3.1-8B-Instruct",
            "vibe_summary",
            "A great model.",
            "str",
        )
        populated_conn.commit()
        missing = _find_models_without_vibes(
            populated_conn,
            ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-Coder-1.5B"],
        )
        assert "meta-llama/Llama-3.1-8B-Instruct" not in missing
        assert "Qwen/Qwen2.5-Coder-1.5B" in missing

    def test_empty_list(self, populated_conn):
        assert _find_models_without_vibes(populated_conn, []) == []
