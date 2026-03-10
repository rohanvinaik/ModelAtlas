"""Tests for D3 healing orchestration."""

from __future__ import annotations

import json
import sqlite3

import pytest

from model_atlas import db
from model_atlas.phase_d_heal import (
    build_healing_prompt,
    export_d3,
    merge_d3,
    select_healing_candidates,
)


@pytest.fixture
def network_conn():
    """In-memory network database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    db.init_db(conn)
    return conn


def _add_model(conn, model_id, audit_score=0.5, downloads=100):
    db.insert_model(conn, model_id, author="test")
    db.set_metadata(conn, model_id, "audit_score", str(audit_score), "float")
    db.set_metadata(conn, model_id, "downloads", str(downloads), "int")
    db.set_metadata(conn, model_id, "pipeline_tag", "text-generation", "str")


def _add_c2_anchor(conn, model_id, label, bank="CAPABILITY"):
    anchor_id = db.get_or_create_anchor(conn, label, bank, source="c2")
    db.link_anchor(conn, model_id, anchor_id, confidence=0.5)


class TestSelectHealingCandidates:
    def test_local_tier_selects_low_audit_score(self, network_conn):
        """Local tier selects models with audit_score below threshold."""
        _add_model(network_conn, "test/bad", audit_score=0.3)
        _add_model(network_conn, "test/ok", audit_score=0.8)

        selected, meta = select_healing_candidates(
            network_conn, None, "local", budget=10
        )
        assert "test/bad" in selected
        assert "test/ok" not in selected
        assert meta["tier"] == "local"

    def test_local_tier_respects_budget(self, network_conn):
        """Local tier caps at budget."""
        for i in range(10):
            _add_model(network_conn, f"test/model-{i}", audit_score=0.2)

        selected, meta = select_healing_candidates(
            network_conn, None, "local", budget=3
        )
        assert len(selected) == 3

    def test_claude_tier_selects_by_downloads(self, network_conn):
        """Claude tier prioritizes high-download models."""
        _add_model(network_conn, "test/popular", audit_score=0.3, downloads=10000)
        _add_model(network_conn, "test/unpopular", audit_score=0.3, downloads=10)

        selected, meta = select_healing_candidates(
            network_conn, None, "claude", budget=1
        )
        assert len(selected) == 1
        assert meta["tier"] == "claude"

    def test_reproducible_with_seed(self, network_conn):
        """Same seed produces same selection."""
        for i in range(20):
            _add_model(
                network_conn, f"test/model-{i}", audit_score=0.3, downloads=i * 100
            )

        sel1, _ = select_healing_candidates(
            network_conn, None, "local", budget=5, seed=42
        )
        sel2, _ = select_healing_candidates(
            network_conn, None, "local", budget=5, seed=42
        )
        assert sel1 == sel2

    def test_invalid_tier_raises(self, network_conn):
        """Unknown tier raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tier"):
            select_healing_candidates(network_conn, None, "invalid", budget=10)


class TestBuildHealingPrompt:
    def test_includes_model_info(self):
        """Prompt includes model_id and raw metadata."""
        prompt = build_healing_prompt(
            model_id="org/test-model",
            raw_json={
                "author": "org",
                "pipeline_tag": "text-generation",
                "tags": ["code"],
            },
            card_excerpt="A code generation model",
            current_anchors=[
                {"label": "chat", "bank": "CAPABILITY", "confidence": 0.5}
            ],
            audit_findings=[
                {
                    "type": "contradiction",
                    "bank": "CAPABILITY",
                    "c2_anchor": "chat",
                    "det_anchor": "code-generation",
                }
            ],
            capability_anchors=["chat", "code-generation", "reasoning"],
            domain_anchors=["code-domain", "math-domain"],
        )
        assert "org/test-model" in prompt
        assert "text-generation" in prompt
        assert "code-generation" in prompt
        assert "contradiction" in prompt

    def test_includes_valid_dictionary(self):
        """Prompt includes valid anchor dictionary."""
        prompt = build_healing_prompt(
            model_id="test/m",
            raw_json={},
            card_excerpt="",
            current_anchors=[],
            audit_findings=[],
            capability_anchors=["reasoning", "chat"],
            domain_anchors=["code-domain"],
        )
        assert "reasoning" in prompt
        assert "code-domain" in prompt


class TestExportD3:
    def test_exports_shards(self, network_conn, tmp_path, monkeypatch):
        """export_d3 creates sharded JSONL files."""
        monkeypatch.setattr("model_atlas.phase_d_heal.PHASE_D_WORK_DIR", tmp_path)

        _add_model(network_conn, "test/model-a", audit_score=0.3)
        _add_c2_anchor(network_conn, "test/model-a", "chat")
        db.set_metadata(
            network_conn, "test/model-a", "qwen_summary", "A chat model", "str"
        )

        result = export_d3(
            network_conn, None, tier="local", budget=10, num_shards=1, seed=42
        )
        assert result.total_exported == 1
        assert result.run_id

        shard = (tmp_path / "d3_local_shard_0.jsonl").read_text().strip()
        item = json.loads(shard)
        assert item["model_id"] == "test/model-a"
        assert "healing_prompt" in item
        assert "valid_anchors" in item
        assert "original_response" in item

    def test_no_candidates_returns_empty(self, network_conn, tmp_path, monkeypatch):
        """No healing candidates → empty export."""
        monkeypatch.setattr("model_atlas.phase_d_heal.PHASE_D_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/good", audit_score=0.9)

        result = export_d3(network_conn, None, tier="local", budget=10)
        assert result.total_exported == 0


class TestMergeD3:
    def test_merges_healed_results(self, network_conn, tmp_path):
        """merge_d3 applies anchor changes and stores correction events."""
        _add_model(network_conn, "test/model-a")
        db.get_or_create_anchor(network_conn, "reasoning", "CAPABILITY", source="seed")
        _add_c2_anchor(network_conn, "test/model-a", "chat")

        run_id = db.create_phase_d_run(network_conn, "d3a")
        network_conn.commit()

        f = tmp_path / "d3_results.jsonl"
        f.write_text(
            json.dumps(
                {
                    "model_id": "test/model-a",
                    "summary": "A reasoning model",
                    "selected_anchors": ["reasoning"],
                    "rationale": "Fixed from chat to reasoning",
                    "original_response": json.dumps(
                        {"summary": "A chat model", "selected_anchors": ["chat"]}
                    ),
                    "original_prompt": "test prompt",
                }
            )
            + "\n"
        )

        result = merge_d3(network_conn, [str(f)], run_id)
        assert result["merged"] == 1
        assert result["anchors_added"] == 1

        # Check correction event stored
        event = network_conn.execute(
            "SELECT rationale FROM correction_events WHERE model_id = 'test/model-a'"
        ).fetchone()
        assert event[0] == "Fixed from chat to reasoning"

    def test_skips_errors(self, network_conn, tmp_path):
        """Error records are skipped."""
        run_id = db.create_phase_d_run(network_conn, "d3a")
        network_conn.commit()

        f = tmp_path / "d3_results.jsonl"
        f.write_text(
            json.dumps(
                {
                    "model_id": "test/model-a",
                    "error": "parse failed",
                }
            )
            + "\n"
        )

        result = merge_d3(network_conn, [str(f)], run_id)
        assert result["skipped"] == 1
        assert result["merged"] == 0

    def test_removes_c2_anchors(self, network_conn, tmp_path):
        """merge_d3 removes C2 anchors not in healed set."""
        _add_model(network_conn, "test/model-a")
        _add_c2_anchor(network_conn, "test/model-a", "chat")
        db.get_or_create_anchor(network_conn, "reasoning", "CAPABILITY", source="seed")

        run_id = db.create_phase_d_run(network_conn, "d3a")
        network_conn.commit()

        f = tmp_path / "d3_results.jsonl"
        f.write_text(
            json.dumps(
                {
                    "model_id": "test/model-a",
                    "summary": "A reasoning model",
                    "selected_anchors": ["reasoning"],
                    "original_response": json.dumps(
                        {"summary": "old", "selected_anchors": ["chat"]}
                    ),
                    "original_prompt": "test prompt",
                }
            )
            + "\n"
        )

        result = merge_d3(network_conn, [str(f)], run_id)
        assert result["anchors_removed"] >= 1

    def test_run_record_updated(self, network_conn, tmp_path):
        """merge_d3 updates the run record."""
        run_id = db.create_phase_d_run(network_conn, "d3a")
        network_conn.commit()

        f = tmp_path / "d3_results.jsonl"
        f.write_text("")

        merge_d3(network_conn, [str(f)], run_id)

        row = network_conn.execute(
            "SELECT status FROM phase_d_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row[0] == "completed"
