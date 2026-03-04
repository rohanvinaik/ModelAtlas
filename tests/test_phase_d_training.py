"""Tests for D4 training data export."""

from __future__ import annotations

import json
import sqlite3

import pytest

from model_atlas import db
from model_atlas.phase_d_training import export_training_data, get_training_data_stats


@pytest.fixture
def network_conn():
    """In-memory network database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    db.init_db(conn)
    return conn


def _add_correction(conn, model_id, tier="local", run_id="test-run"):
    """Insert a correction event with full data."""
    db.insert_model(conn, model_id)
    # Ensure run exists
    try:
        conn.execute(
            "INSERT OR IGNORE INTO phase_d_runs (run_id, phase, started_at, status) VALUES (?, 'd3a', '2024-01-01', 'completed')",
            (run_id,),
        )
    except Exception:
        pass
    db.insert_correction_event(
        conn,
        run_id=run_id,
        model_id=model_id,
        tier=tier,
        original_prompt="Classify this model",
        original_response=json.dumps({"summary": "old", "selected_anchors": ["chat"]}),
        healed_response=json.dumps({"summary": "new", "selected_anchors": ["reasoning"]}),
        anchors_added=["reasoning"],
        anchors_removed=["chat"],
        rationale="Fixed classification",
    )
    conn.commit()


class TestExportTrainingData:
    def test_exports_dpo_format(self, network_conn, tmp_path):
        """Exports DPO-format JSONL with prompt/chosen/rejected."""
        _add_correction(network_conn, "test/model-a")

        output = tmp_path / "training.jsonl"
        stats = export_training_data(network_conn, output)

        assert stats.total_examples == 1
        assert stats.output_path == str(output)

        line = output.read_text().strip()
        item = json.loads(line)
        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item
        assert item["model_id"] == "test/model-a"
        assert item["tier"] == "local"

    def test_skips_identical_responses(self, network_conn, tmp_path):
        """Skips corrections where original == healed."""
        db.insert_model(network_conn, "test/model-a")
        conn = network_conn
        conn.execute(
            "INSERT OR IGNORE INTO phase_d_runs (run_id, phase, started_at, status) VALUES ('r1', 'd3a', '2024-01-01', 'completed')"
        )
        same_response = json.dumps({"summary": "same", "selected_anchors": ["chat"]})
        db.insert_correction_event(
            conn,
            run_id="r1",
            model_id="test/model-a",
            tier="local",
            original_prompt="prompt",
            original_response=same_response,
            healed_response=same_response,
        )
        conn.commit()

        output = tmp_path / "training.jsonl"
        stats = export_training_data(network_conn, output)
        assert stats.total_examples == 0

    def test_filters_by_tier(self, network_conn, tmp_path):
        """tier parameter filters output."""
        _add_correction(network_conn, "test/model-a", tier="local", run_id="r1")
        _add_correction(network_conn, "test/model-b", tier="claude", run_id="r2")

        output = tmp_path / "local_only.jsonl"
        stats = export_training_data(network_conn, output, tier="local")
        assert stats.total_examples == 1
        assert stats.by_tier == {"local": 1}

    def test_all_tier(self, network_conn, tmp_path):
        """tier='all' exports everything."""
        _add_correction(network_conn, "test/model-a", tier="local", run_id="r1")
        _add_correction(network_conn, "test/model-b", tier="claude", run_id="r2")

        output = tmp_path / "all.jsonl"
        stats = export_training_data(network_conn, output, tier="all")
        assert stats.total_examples == 2

    def test_default_output_path(self, network_conn, monkeypatch, tmp_path):
        """Uses default path when output_path is None."""
        monkeypatch.setattr("model_atlas.phase_d_training.PHASE_D_TRAINING_DIR", tmp_path)
        _add_correction(network_conn, "test/model-a")

        stats = export_training_data(network_conn)
        assert stats.total_examples == 1
        assert "dpo_training.jsonl" in stats.output_path

    def test_empty_db(self, network_conn, tmp_path):
        """Empty correction_events → 0 examples."""
        output = tmp_path / "training.jsonl"
        stats = export_training_data(network_conn, output)
        assert stats.total_examples == 0


class TestGetTrainingDataStats:
    def test_returns_counts(self, network_conn):
        """Returns correct counts."""
        _add_correction(network_conn, "test/model-a", tier="local")
        _add_correction(network_conn, "test/model-b", tier="claude", run_id="r2")

        stats = get_training_data_stats(network_conn)
        assert stats["total_corrections"] == 2
        assert stats["distinct_models"] == 2
        assert stats["by_tier"]["local"] == 1
        assert stats["by_tier"]["claude"] == 1

    def test_empty_db(self, network_conn):
        """Empty DB returns zeros."""
        stats = get_training_data_stats(network_conn)
        assert stats["total_corrections"] == 0
        assert stats["distinct_models"] == 0
