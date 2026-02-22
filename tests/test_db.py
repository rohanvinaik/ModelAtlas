"""Tests for the SQLite database layer."""

from __future__ import annotations

from hf_model_search import db


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
