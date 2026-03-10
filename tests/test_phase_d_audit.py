"""Tests for D1 deterministic audit."""

from __future__ import annotations

import json
import sqlite3

import pytest

from model_atlas import db
from model_atlas.phase_d_audit import audit_c2


@pytest.fixture
def network_conn():
    """In-memory network database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    db.init_db(conn)
    return conn


def _add_model(conn, model_id, author="test", pipeline_tag="text-generation"):
    db.insert_model(conn, model_id, author=author)
    if pipeline_tag:
        db.set_metadata(conn, model_id, "pipeline_tag", pipeline_tag, "str")


def _add_c2_anchor(conn, model_id, label, bank="CAPABILITY"):
    """Link a model to an anchor at C2 confidence (0.5)."""
    anchor_id = db.get_or_create_anchor(conn, label, bank, source="c2")
    db.link_anchor(conn, model_id, anchor_id, confidence=0.5)


class TestAuditC2:
    def test_no_c2_models(self, network_conn):
        """No C2 anchors → 0 audited."""
        _add_model(network_conn, "test/model-a")
        result = audit_c2(network_conn)
        assert result.total_audited == 0
        assert result.total_mismatches == 0

    def test_clean_model_no_findings(self, network_conn):
        """Model with C2 anchor matching pattern signals → no contradiction."""
        # model_id contains "code" → patterns will match code-generation
        _add_model(network_conn, "test/code-model", pipeline_tag="text-generation")
        # C2 assigned code-generation — matches what patterns find from "code" in name
        _add_c2_anchor(network_conn, "test/code-model", "code-generation", "CAPABILITY")

        result = audit_c2(network_conn)
        assert result.total_audited == 1
        # No contradiction since C2 agrees with deterministic
        assert result.per_type_counts.get("contradiction", 0) == 0

    def test_contradiction_detected(self, network_conn):
        """C2 assigned wrong anchor vs pattern re-run → contradiction."""
        # "medical" in name triggers medical-domain pattern, but C2 assigned code-domain
        _add_model(network_conn, "test/medical-model", pipeline_tag="text-generation")
        _add_c2_anchor(network_conn, "test/medical-model", "code-domain", "DOMAIN")

        result = audit_c2(network_conn)
        assert result.total_mismatches >= 1
        assert result.per_type_counts.get("contradiction", 0) >= 1

    def test_gap_detected(self, network_conn):
        """Pattern re-run finds anchor that C2 missed → gap."""
        # "instruct" in name triggers instruction-following pattern
        _add_model(network_conn, "test/instruct-model", pipeline_tag="text-generation")
        # C2 assigned chat but missed instruction-following
        _add_c2_anchor(network_conn, "test/instruct-model", "chat", "CAPABILITY")

        result = audit_c2(network_conn)
        # "instruct" triggers instruction-following, but C2 only has "chat"
        # → gap for instruction-following
        assert result.total_mismatches >= 1

    def test_confidence_conflict_detected(self, network_conn, monkeypatch):
        """Same anchor at very different confidences → confidence_conflict."""
        from model_atlas import phase_d_audit

        _add_model(network_conn, "test/model-a")
        label = "code-generation"
        anchor_id = db.get_or_create_anchor(network_conn, label, "CAPABILITY")
        db.link_anchor(network_conn, "test/model-a", anchor_id, confidence=0.5)

        # The confidence_conflict path compares _get_c2_anchors (conf=0.5) vs
        # _get_det_anchors (conf>=0.8) for the same label. With INSERT OR REPLACE
        # on (model_id, anchor_id), one row can't hold two confidences.
        # Patch _get_det_anchors to return a high-confidence match for this label.
        def _fake_det_anchors(conn, model_id):
            return {"CAPABILITY": [(label, 0.9)]}

        monkeypatch.setattr(phase_d_audit, "_get_det_anchors", _fake_det_anchors)

        result = audit_c2(network_conn)
        assert result.per_type_counts.get("confidence_conflict", 0) >= 1

    def test_audit_score_stored(self, network_conn):
        """Audit stores per-model audit_score in metadata."""
        _add_model(network_conn, "test/model-a")
        _add_c2_anchor(network_conn, "test/model-a", "code-generation", "CAPABILITY")

        audit_c2(network_conn)

        row = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'audit_score'"
        ).fetchone()
        assert row is not None
        score = float(row[0])
        assert 0.0 <= score <= 1.0

    def test_run_record_created(self, network_conn):
        """Audit creates a phase_d_runs record."""
        _add_model(network_conn, "test/model-a")
        _add_c2_anchor(network_conn, "test/model-a", "chat", "CAPABILITY")

        result = audit_c2(network_conn)

        row = network_conn.execute(
            "SELECT phase, status FROM phase_d_runs WHERE run_id = ?",
            (result.run_id,),
        ).fetchone()
        assert row[0] == "d1"
        assert row[1] == "completed"

    def test_findings_stored_in_db(self, network_conn):
        """Audit findings are persisted in audit_findings table."""
        # "medical" triggers medical-domain, but C2 has code-domain → contradiction
        _add_model(network_conn, "test/medical-llm", pipeline_tag="text-generation")
        _add_c2_anchor(network_conn, "test/medical-llm", "code-domain", "DOMAIN")

        result = audit_c2(network_conn)

        count = network_conn.execute(
            "SELECT COUNT(*) FROM audit_findings WHERE run_id = ?",
            (result.run_id,),
        ).fetchone()[0]
        assert count > 0

    def test_with_ingest_conn(self, network_conn):
        """Audit works with ingest_conn for raw_json access."""
        ingest_conn = sqlite3.connect(":memory:")
        ingest_conn.row_factory = sqlite3.Row
        ingest_conn.execute(
            """CREATE TABLE ingest_models (
                model_id TEXT PRIMARY KEY,
                raw_json TEXT
            )"""
        )
        ingest_conn.execute(
            "INSERT INTO ingest_models (model_id, raw_json) VALUES (?, ?)",
            (
                "test/model-a",
                json.dumps({"tags": ["code"], "pipeline_tag": "text-generation"}),
            ),
        )
        ingest_conn.commit()

        _add_model(network_conn, "test/model-a")
        _add_c2_anchor(network_conn, "test/model-a", "chat", "CAPABILITY")

        result = audit_c2(network_conn, ingest_conn)
        assert result.total_audited == 1
        ingest_conn.close()
