"""Tests for `spreading_activation` — two-layer activation over models + anchors."""

from __future__ import annotations

from model_atlas import db, spreading_activation as sa


def _seed(conn) -> None:
    for mid in ("src", "kid", "sibling", "unrelated"):
        db.insert_model(conn, mid, author="test")
    # Layer 1: lineage edge — kid fine-tuned from src
    db.add_link(conn, "kid", "src", "fine_tuned_from")
    # Layer 2: sibling shares an anchor with src (no direct link)
    aid = db.get_or_create_anchor(conn, "shared-tag", "CAPABILITY")
    db.link_anchor(conn, "src", aid, confidence=0.9)
    db.link_anchor(conn, "sibling", aid, confidence=0.9)
    conn.commit()


def test_layer1_reaches_lineage_neighbors(conn):
    _seed(conn)
    results = sa.spread(conn, "src")
    reached = {r.model_id for r in results}
    assert "kid" in reached


def test_layer2_reaches_anchor_siblings(conn):
    """`sibling` has no direct edge to `src`, only a shared anchor. It
    should appear only via layer 2, and its `bank_activations` should
    be non-empty for the anchor's bank (CAPABILITY)."""
    _seed(conn)
    results = sa.spread(conn, "src")
    reached = {r.model_id: r for r in results}
    assert "sibling" in reached
    banks = reached["sibling"].bank_activations
    assert "CAPABILITY" in banks
    assert banks["CAPABILITY"] > 0


def test_source_excluded_from_results(conn):
    _seed(conn)
    results = sa.spread(conn, "src")
    reached = {r.model_id for r in results}
    assert "src" not in reached


def test_unrelated_not_reached(conn):
    _seed(conn)
    results = sa.spread(conn, "src")
    reached = {r.model_id for r in results}
    assert "unrelated" not in reached


def test_use_anchors_false_disables_layer2(conn):
    """When the anchor layer is off, sibling — reachable only via anchor —
    should drop out. `kid` still reached via layer 1."""
    _seed(conn)
    results = sa.spread(conn, "src", config=sa.SpreadingConfig(use_anchors=False))
    reached = {r.model_id for r in results}
    assert "kid" in reached
    assert "sibling" not in reached


def test_popularity_cutoff_prunes_generic_anchors(conn):
    """An anchor attached to thousands of models is generic — activating
    every one drowns the signal. The cutoff should skip it entirely."""
    db.insert_model(conn, "src", author="test")
    db.insert_model(conn, "gen1", author="test")
    db.insert_model(conn, "gen2", author="test")
    aid = db.get_or_create_anchor(conn, "generic-anchor", "CAPABILITY")
    db.link_anchor(conn, "src", aid, confidence=0.9)
    db.link_anchor(conn, "gen1", aid, confidence=0.9)
    db.link_anchor(conn, "gen2", aid, confidence=0.9)
    conn.commit()
    # Cutoff=1 → any anchor covering >1 model is generic, skipped.
    results = sa.spread(
        conn, "src", config=sa.SpreadingConfig(anchor_popularity_cutoff=1)
    )
    reached = {r.model_id for r in results}
    assert "gen1" not in reached and "gen2" not in reached


def test_anchor_neighbors_bank_filter(conn):
    """`anchor_neighbors(bank='CAPABILITY')` filters to that bank only,
    even when the source has anchors in other banks that would reach
    other siblings."""
    db.insert_model(conn, "src", author="test")
    db.insert_model(conn, "cap_sib", author="test")
    db.insert_model(conn, "arch_sib", author="test")
    a_cap = db.get_or_create_anchor(conn, "cap-anchor", "CAPABILITY")
    a_arch = db.get_or_create_anchor(conn, "arch-anchor", "ARCHITECTURE")
    for m in ("src", "cap_sib"):
        db.link_anchor(conn, m, a_cap, confidence=0.9)
    for m in ("src", "arch_sib"):
        db.link_anchor(conn, m, a_arch, confidence=0.9)
    conn.commit()
    cap_only = sa.anchor_neighbors(conn, "src", bank="CAPABILITY")
    reached = {r[0] for r in cap_only}
    assert "cap_sib" in reached
    assert "arch_sib" not in reached


def test_multi_source_union(conn):
    """Multi-source spread activates the union of neighborhoods. If a
    model is reachable from EITHER source, it appears."""
    db.insert_model(conn, "s1", author="test")
    db.insert_model(conn, "s2", author="test")
    db.insert_model(conn, "child_of_s1", author="test")
    db.insert_model(conn, "child_of_s2", author="test")
    db.add_link(conn, "child_of_s1", "s1", "fine_tuned_from")
    db.add_link(conn, "child_of_s2", "s2", "fine_tuned_from")
    conn.commit()
    results = sa.spread(conn, ["s1", "s2"])
    reached = {r.model_id for r in results}
    assert "child_of_s1" in reached
    assert "child_of_s2" in reached
