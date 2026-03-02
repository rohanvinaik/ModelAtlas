"""Tests for MCP server tools with mocked DB."""

from __future__ import annotations

import json

from model_atlas import db
from model_atlas.search.structured import StructuredResult
from model_atlas.server import (
    _build_index_huggingface,
    _find_models_without_vibes,
    hf_compare_models,
    hf_get_model_detail,
    hf_index_status,
    hf_search_models,
    navigate_models,
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


class TestNavigateModels:
    def test_empty_network_returns_error(self, conn, monkeypatch):
        _patch_db(monkeypatch, conn)
        result = json.loads(navigate_models())
        assert "error" in result
        assert result["network_models"] == 0

    def test_returns_results_with_populated_db(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(navigate_models(efficiency=-1))
        assert result["network_models"] == 4
        assert result["result_count"] >= 1
        assert "results" in result

    def test_bank_directions_in_query(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(navigate_models(capability=1, domain=1))
        assert result["query"]["banks"]["CAPABILITY"] == 1
        assert result["query"]["banks"]["DOMAIN"] == 1

    def test_require_anchors_filter(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(navigate_models(require_anchors=["code-generation"]))
        assert result["query"]["require_anchors"] == ["code-generation"]
        # Only code model should match
        model_ids = [r["model_id"] for r in result["results"]]
        assert "Qwen/Qwen2.5-Coder-1.5B" in model_ids

    def test_prefer_anchors_boost(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(navigate_models(prefer_anchors=["medical-domain"]))
        assert result["query"]["prefer_anchors"] == ["medical-domain"]
        assert result["result_count"] >= 1

    def test_avoid_anchors_penalty(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(navigate_models(avoid_anchors=["quantized"]))
        assert result["query"]["avoid_anchors"] == ["quantized"]

    def test_similar_to(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(
            navigate_models(similar_to="meta-llama/Llama-3.1-8B-Instruct")
        )
        assert result["query"]["similar_to"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert result["result_count"] >= 1

    def test_limit_respected(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(navigate_models(limit=2))
        assert result["result_count"] <= 2

    def test_result_structure(self, populated_conn, monkeypatch):
        _patch_db(monkeypatch, populated_conn)
        result = json.loads(navigate_models())
        if result["results"]:
            r = result["results"][0]
            assert "model_id" in r
            assert "score" in r
            assert "score_breakdown" in r
            assert "bank_alignment" in r["score_breakdown"]
            assert "anchor_relevance" in r["score_breakdown"]
            assert "seed_similarity" in r["score_breakdown"]
            assert "positions" in r
            assert "anchors" in r


def _make_structured_result(model_id="test/Model", rank=0):
    """Create a mock StructuredResult for testing."""
    return StructuredResult(
        model_id=model_id,
        author=model_id.split("/")[0] if "/" in model_id else "",
        likes=100,
        downloads=5000,
        pipeline_tag="text-generation",
        tags=["text-generation"],
        library_name="transformers",
        license="apache-2.0",
        card_text="A test model.",
        rank=rank,
        raw={"created_at": "2025-01-01"},
    )


class TestHfSearchModels:
    def _patch_search(self, monkeypatch, conn, candidates=None):
        """Patch both DB and structured search for hf_search_models."""
        _patch_db(monkeypatch, conn)
        if candidates is None:
            candidates = [_make_structured_result("test/A", 0)]
        monkeypatch.setattr(
            "model_atlas.server.structured.search", lambda **kw: candidates
        )

    def test_empty_network_returns_fuzzy_fallback(self, conn, monkeypatch):
        self._patch_search(monkeypatch, conn)
        result = json.loads(hf_search_models("test query"))
        assert result["network_status"] == "empty"
        assert "hint" in result
        assert result["total_candidates"] == 1

    def test_populated_network_returns_network_results(
        self, populated_conn, monkeypatch
    ):
        self._patch_search(monkeypatch, populated_conn)
        result = json.loads(hf_search_models("instruct model"))
        assert result["network_status"] == "active"
        assert result["network_models"] == 4
        assert "hint" not in result

    def test_filters_passed_through(self, conn, monkeypatch):
        self._patch_search(monkeypatch, conn)
        result = json.loads(
            hf_search_models(
                "code model", task="text-generation", author="meta", library="gguf"
            )
        )
        assert result["filters"]["task"] == "text-generation"
        assert result["filters"]["author"] == "meta"
        assert result["filters"]["library"] == "gguf"

    def test_empty_candidates(self, conn, monkeypatch):
        self._patch_search(monkeypatch, conn, candidates=[])
        result = json.loads(hf_search_models("nonexistent"))
        assert result["total_candidates"] == 0
        assert result["result_count"] == 0

    def test_output_structure(self, conn, monkeypatch):
        self._patch_search(monkeypatch, conn)
        result = json.loads(hf_search_models("test"))
        assert "query" in result
        assert "filters" in result
        assert "network_status" in result
        assert "total_candidates" in result
        assert "result_count" in result
        assert "results" in result


class TestBuildIndexHuggingface:
    def test_no_candidates_returns_error(self, conn, monkeypatch):
        monkeypatch.setattr(
            "model_atlas.server.structured.search", lambda **kw: []
        )
        result = json.loads(_build_index_huggingface(conn, "code", None, 100, 5))
        assert "error" in result

    def test_indexes_candidates(self, conn, monkeypatch):
        db.init_db(conn)
        candidates = [
            _make_structured_result("test/A", 0),
            _make_structured_result("test/B", 1),
        ]
        monkeypatch.setattr(
            "model_atlas.server.structured.search", lambda **kw: candidates
        )
        result = json.loads(_build_index_huggingface(conn, "text-gen", None, 100, 5))
        assert result["status"] == "indexed"
        assert result["models_fetched"] == 2
        assert result["models_indexed"] == 2
        assert result["network_total"] >= 2

    def test_vibes_needed_reported(self, conn, monkeypatch):
        db.init_db(conn)
        candidates = [_make_structured_result("test/A", 0)]
        monkeypatch.setattr(
            "model_atlas.server.structured.search", lambda **kw: candidates
        )
        result = json.loads(_build_index_huggingface(conn, "text-gen", None, 100, 5))
        assert "vibes_needed" in result
        assert "test/A" in result["vibes_needed"]
