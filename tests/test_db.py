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
