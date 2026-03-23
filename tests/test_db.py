"""Tests for the SQLite database layer."""

from __future__ import annotations

import sqlite3

from model_atlas import db


def test_init_creates_tables(conn):
    """Schema creation produces all 6 tables."""
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {t["name"] for t in tables}
    assert "models" in names
    assert "model_positions" in names
    assert "model_links" in names
    assert "anchors" in names
    assert "model_anchors" in names
    assert "model_metadata" in names


def test_bootstrap_anchors(conn):
    """Bootstrap populates the anchor dictionary."""
    count = conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
    assert count > 50  # We bootstrap ~90 anchors


def test_insert_and_get_model(conn):
    """Insert a model and retrieve it with full profile."""
    db.insert_model(conn, "test/Model-7B", author="test", source="huggingface")
    db.set_position(conn, "test/Model-7B", "ARCHITECTURE", 0, 0, ["decoder-only"])
    db.set_position(conn, "test/Model-7B", "EFFICIENCY", 0, 0)
    db.set_metadata(conn, "test/Model-7B", "license", "apache-2.0", "str")
    conn.commit()

    model = db.get_model(conn, "test/Model-7B")
    assert model is not None
    assert model["model_id"] == "test/Model-7B"
    assert model["author"] == "test"
    assert "ARCHITECTURE" in model["positions"]
    assert model["positions"]["ARCHITECTURE"]["sign"] == 0
    assert model["metadata"]["license"]["value"] == "apache-2.0"


def test_anchor_linking(conn):
    """Models can be linked to anchors and queried back."""
    db.insert_model(conn, "test/Model-A", author="test")
    aid = db.get_or_create_anchor(conn, "code-generation", "CAPABILITY", "skill")
    db.link_anchor(conn, "test/Model-A", aid)
    conn.commit()

    anchors = db.get_anchor_set(conn, "test/Model-A")
    assert "code-generation" in anchors

    models = db.find_models_by_anchor(conn, "code-generation")
    assert "test/Model-A" in models


def test_model_links(conn):
    """Explicit relationships between models."""
    db.insert_model(conn, "base/Model", author="base")
    db.insert_model(conn, "derived/Model-FT", author="derived")
    db.add_link(conn, "derived/Model-FT", "base/Model", "fine_tuned_from")
    conn.commit()

    model = db.get_model(conn, "derived/Model-FT")
    assert model is not None
    outgoing = model["links"]["outgoing"]
    assert len(outgoing) == 1
    assert outgoing[0]["target_id"] == "base/Model"
    assert outgoing[0]["relation"] == "fine_tuned_from"


def test_bank_range_query(conn):
    """Find models within a bank position range."""
    db.insert_model(conn, "small/Model-1B", author="test")
    db.set_position(conn, "small/Model-1B", "EFFICIENCY", -1, 2)
    db.insert_model(conn, "big/Model-70B", author="test")
    db.set_position(conn, "big/Model-70B", "EFFICIENCY", 1, 3)
    conn.commit()

    small = db.find_models_by_bank_range(conn, "EFFICIENCY", max_signed=-1)
    assert any(m["model_id"] == "small/Model-1B" for m in small)
    assert not any(m["model_id"] == "big/Model-70B" for m in small)


def test_network_stats(conn):
    """Stats reflect the network state."""
    db.insert_model(conn, "test/Model", author="test")
    db.set_position(conn, "test/Model", "ARCHITECTURE", 0, 0)
    conn.commit()

    stats = db.network_stats(conn)
    assert stats["total_models"] == 1
    assert stats["total_anchors"] > 0
    assert stats["models_per_bank"]["ARCHITECTURE"] == 1


def test_upsert_model(conn):
    """Inserting the same model twice updates it."""
    db.insert_model(conn, "test/Model", author="old")
    db.insert_model(conn, "test/Model", author="new")
    conn.commit()

    model = db.get_model(conn, "test/Model")
    assert model is not None
    assert model["author"] == "new"


def test_transaction_rollback_on_error(conn):
    """Transaction rolls back on exception."""
    db.insert_model(conn, "test/Rollback", author="test")
    conn.commit()

    try:
        with db.transaction(conn) as c:
            db.insert_model(c, "test/Rollback", author="changed")
            raise ValueError("simulated error")
    except ValueError:
        pass

    model = db.get_model(conn, "test/Rollback")
    assert model is not None
    assert model["author"] == "test"  # rolled back to original


def test_anchor_bank_reassignment_warning(conn, caplog):
    """Creating an anchor in a different bank logs a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        aid1 = db.get_or_create_anchor(conn, "code-gen", "CAPABILITY", "skill")
        aid2 = db.get_or_create_anchor(conn, "code-gen", "DOMAIN", "skill")
    # Same anchor ID returned
    assert aid1 == aid2
    assert "already assigned to bank" in caplog.text


def test_set_position_with_path_nodes(conn):
    """Path nodes are stored as JSON."""
    db.insert_model(conn, "test/Model", author="test")
    db.set_position(conn, "test/Model", "ARCHITECTURE", 1, 2, ["MoE", "sparse"])
    conn.commit()

    model = db.get_model(conn, "test/Model")
    assert model is not None
    arch = model["positions"]["ARCHITECTURE"]
    assert arch["sign"] == 1
    assert arch["depth"] == 2


def test_link_anchor_with_confidence(conn):
    """Anchors can be linked with custom weight and confidence."""
    db.insert_model(conn, "test/Model", author="test")
    aid = db.get_or_create_anchor(conn, "test-anchor", "CAPABILITY")
    db.link_anchor(conn, "test/Model", aid, weight=0.8, confidence=0.6)
    conn.commit()

    row = conn.execute(
        "SELECT weight, confidence FROM model_anchors WHERE model_id = ? AND anchor_id = ?",
        ("test/Model", aid),
    ).fetchone()
    assert abs(row["weight"] - 0.8) < 0.001
    assert abs(row["confidence"] - 0.6) < 0.001


def test_batch_get_positions(conn):
    """Batch position retrieval returns data for multiple models."""
    for mid in ["test/A", "test/B"]:
        db.insert_model(conn, mid, author="test")
        db.set_position(conn, mid, "EFFICIENCY", 0, 0)
    conn.commit()

    positions = db.batch_get_positions(conn, ["test/A", "test/B"])
    assert "test/A" in positions
    assert "test/B" in positions
    assert "EFFICIENCY" in positions["test/A"]


def test_batch_get_anchor_sets(conn):
    """Batch anchor set retrieval works for multiple models."""
    for mid in ["test/A", "test/B"]:
        db.insert_model(conn, mid, author="test")
        aid = db.get_or_create_anchor(conn, f"{mid}-anchor", "CAPABILITY")
        db.link_anchor(conn, mid, aid)
    conn.commit()

    anchor_sets = db.batch_get_anchor_sets(conn, ["test/A", "test/B"])
    assert "test/A-anchor" in anchor_sets["test/A"]
    assert "test/B-anchor" in anchor_sets["test/B"]


def test_compute_anchor_idf(conn):
    """IDF computation returns positive values for rare anchors."""
    db.insert_model(conn, "test/A", author="test")
    db.insert_model(conn, "test/B", author="test")
    rare = db.get_or_create_anchor(conn, "rare-anchor", "CAPABILITY")
    common = db.get_or_create_anchor(conn, "common-anchor", "CAPABILITY")
    db.link_anchor(conn, "test/A", rare)
    db.link_anchor(conn, "test/A", common)
    db.link_anchor(conn, "test/B", common)
    conn.commit()

    idf = db.compute_anchor_idf(conn)
    # rare-anchor appears in 1 of 2 models → IDF > 0
    assert "rare-anchor" in idf
    assert idf["rare-anchor"] > 0
    # common-anchor appears in all models → IDF = 0
    assert idf["common-anchor"] == 0.0
    # Rare anchor has higher IDF than common
    assert idf["rare-anchor"] > idf["common-anchor"]


def test_get_connection(tmp_path, monkeypatch):
    """get_connection returns a configured SQLite connection."""
    db_path = tmp_path / "data" / "network.db"
    monkeypatch.setattr("model_atlas.db.NETWORK_DB_PATH", db_path)
    conn = db.get_connection()
    assert isinstance(conn, sqlite3.Connection)
    assert conn.row_factory == sqlite3.Row
    # WAL mode enabled
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    # Foreign keys enabled
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1
    conn.close()
    # Parent dir was created
    assert db_path.parent.exists()


# ---------------------------------------------------------------------------
# Phase D: audit/correction lifecycle
# ---------------------------------------------------------------------------


class TestCreatePhaseDRun:
    """Tests for create_phase_d_run."""

    def test_returns_uuid_string(self, conn):
        run_id = db.create_phase_d_run(conn, "d1", config={"phase": "audit"})
        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID format

    def test_persists_to_database(self, conn):
        run_id = db.create_phase_d_run(conn, "d1")
        row = conn.execute(
            "SELECT phase, status FROM phase_d_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row is not None
        assert row["phase"] == "d1"
        assert row["status"] == "running"

    def test_config_stored_as_json(self, conn):
        config = {"top_k": 10, "threshold": 0.5}
        run_id = db.create_phase_d_run(conn, "d3", config=config)
        import json

        row = conn.execute(
            "SELECT config FROM phase_d_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert json.loads(row["config"]) == config

    def test_null_config(self, conn):
        run_id = db.create_phase_d_run(conn, "d1", config=None)
        row = conn.execute(
            "SELECT config FROM phase_d_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["config"] is None


class TestFinishPhaseDRun:
    """Tests for finish_phase_d_run."""

    def test_updates_status(self, conn):
        run_id = db.create_phase_d_run(conn, "d1")
        db.finish_phase_d_run(conn, run_id, "completed")
        row = conn.execute(
            "SELECT status, finished_at FROM phase_d_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "completed"
        assert row["finished_at"] is not None

    def test_stores_summary_as_json(self, conn):
        import json

        run_id = db.create_phase_d_run(conn, "d1")
        summary = {"total": 100, "mismatches": 5}
        db.finish_phase_d_run(conn, run_id, "completed", summary=summary)
        row = conn.execute(
            "SELECT summary FROM phase_d_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert json.loads(row["summary"]) == summary

    def test_null_summary(self, conn):
        run_id = db.create_phase_d_run(conn, "d1")
        db.finish_phase_d_run(conn, run_id, "failed")
        row = conn.execute(
            "SELECT summary, status FROM phase_d_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["summary"] is None
        assert row["status"] == "failed"


class TestInsertAuditFinding:
    """Tests for insert_audit_finding — the highest-gamma composition hub."""

    def test_returns_finding_id(self, conn):
        run_id = db.create_phase_d_run(conn, "d1")
        db.insert_model(conn, "test/Model", author="test")
        finding_id = db.insert_audit_finding(
            conn,
            run_id=run_id,
            model_id="test/Model",
            mismatch_type="contradiction",
            bank="CAPABILITY",
        )
        assert isinstance(finding_id, int)
        assert finding_id > 0

    def test_persists_all_fields(self, conn):
        import json

        run_id = db.create_phase_d_run(conn, "d1")
        db.insert_model(conn, "test/Model", author="test")
        detail = {"pipeline_tag": "text-generation", "det_found": ["code-gen"]}
        finding_id = db.insert_audit_finding(
            conn,
            run_id=run_id,
            model_id="test/Model",
            mismatch_type="contradiction",
            bank="CAPABILITY",
            c2_anchor="chat",
            det_anchor=None,
            severity=0.7,
            detail=detail,
        )
        row = conn.execute(
            "SELECT * FROM audit_findings WHERE finding_id = ?", (finding_id,)
        ).fetchone()
        assert row["run_id"] == run_id
        assert row["model_id"] == "test/Model"
        assert row["mismatch_type"] == "contradiction"
        assert row["bank"] == "CAPABILITY"
        assert row["c2_anchor"] == "chat"
        assert row["det_anchor"] is None
        assert abs(row["severity"] - 0.7) < 0.001
        assert json.loads(row["detail"]) == detail

    def test_nullable_fields_default_to_none(self, conn):
        run_id = db.create_phase_d_run(conn, "d1")
        db.insert_model(conn, "test/Model", author="test")
        finding_id = db.insert_audit_finding(
            conn,
            run_id=run_id,
            model_id="test/Model",
            mismatch_type="gap",
        )
        row = conn.execute(
            "SELECT bank, c2_anchor, det_anchor, detail FROM audit_findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()
        assert row["bank"] is None
        assert row["c2_anchor"] is None
        assert row["det_anchor"] is None
        assert row["detail"] is None

    def test_default_severity(self, conn):
        run_id = db.create_phase_d_run(conn, "d1")
        db.insert_model(conn, "test/Model", author="test")
        finding_id = db.insert_audit_finding(
            conn,
            run_id=run_id,
            model_id="test/Model",
            mismatch_type="unsupported",
        )
        row = conn.execute(
            "SELECT severity FROM audit_findings WHERE finding_id = ?", (finding_id,)
        ).fetchone()
        assert abs(row["severity"] - 0.5) < 0.001

    def test_multiple_findings_same_run(self, conn):
        run_id = db.create_phase_d_run(conn, "d1")
        db.insert_model(conn, "test/A", author="test")
        db.insert_model(conn, "test/B", author="test")
        id1 = db.insert_audit_finding(
            conn,
            run_id=run_id,
            model_id="test/A",
            mismatch_type="gap",
        )
        id2 = db.insert_audit_finding(
            conn,
            run_id=run_id,
            model_id="test/B",
            mismatch_type="contradiction",
        )
        assert id1 != id2
        count = conn.execute(
            "SELECT COUNT(*) FROM audit_findings WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert count == 2


class TestInsertCorrectionEvent:
    """Tests for insert_correction_event — the other high-gamma hub."""

    def test_returns_event_id(self, conn):
        run_id = db.create_phase_d_run(conn, "d3")
        db.insert_model(conn, "test/Model", author="test")
        event_id = db.insert_correction_event(
            conn,
            run_id=run_id,
            model_id="test/Model",
            tier="d3_heal",
        )
        assert isinstance(event_id, int)
        assert event_id > 0

    def test_persists_all_fields(self, conn):
        import json

        run_id = db.create_phase_d_run(conn, "d3")
        db.insert_model(conn, "test/Model", author="test")
        event_id = db.insert_correction_event(
            conn,
            run_id=run_id,
            model_id="test/Model",
            tier="d3_heal",
            original_prompt="Classify this model",
            original_response='{"anchors": ["chat"]}',
            healed_response='{"anchors": ["chat", "code-gen"]}',
            anchors_added=["code-gen"],
            anchors_removed=[],
            rationale="Model card mentions code generation",
        )
        row = conn.execute(
            "SELECT * FROM correction_events WHERE event_id = ?", (event_id,)
        ).fetchone()
        assert row["run_id"] == run_id
        assert row["model_id"] == "test/Model"
        assert row["tier"] == "d3_heal"
        assert row["original_prompt"] == "Classify this model"
        assert json.loads(row["anchors_added"]) == ["code-gen"]
        # Empty list is falsy → stored as NULL by the current serialization logic
        assert row["anchors_removed"] is None
        assert row["rationale"] == "Model card mentions code generation"
        assert row["created_at"] is not None

    def test_nullable_fields_default_to_none(self, conn):
        run_id = db.create_phase_d_run(conn, "d3")
        db.insert_model(conn, "test/Model", author="test")
        event_id = db.insert_correction_event(
            conn,
            run_id=run_id,
            model_id="test/Model",
            tier="d3_heal",
        )
        row = conn.execute(
            "SELECT original_prompt, original_response, healed_response, "
            "anchors_added, anchors_removed, rationale "
            "FROM correction_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        assert row["original_prompt"] is None
        assert row["original_response"] is None
        assert row["healed_response"] is None
        assert row["anchors_added"] is None
        assert row["anchors_removed"] is None
        assert row["rationale"] is None

    def test_multiple_events_same_run(self, conn):
        run_id = db.create_phase_d_run(conn, "d3")
        db.insert_model(conn, "test/A", author="test")
        db.insert_model(conn, "test/B", author="test")
        id1 = db.insert_correction_event(
            conn,
            run_id=run_id,
            model_id="test/A",
            tier="d3_heal",
        )
        id2 = db.insert_correction_event(
            conn,
            run_id=run_id,
            model_id="test/B",
            tier="d3_heal",
        )
        assert id1 != id2
        count = conn.execute(
            "SELECT COUNT(*) FROM correction_events WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert count == 2
