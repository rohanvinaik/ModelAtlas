"""Tests for `pagerank` — power iteration + attenuated lineage inheritance."""

from __future__ import annotations

from model_atlas import db, pagerank


def _seed_lineage_chain(conn) -> None:
    """A → B (fine-tuned) → C (fine-tuned) → D (quantized).

    All four models share the graph but each has a specific role: A is
    the "base" root that C's rank should flow to; D is a quantized leaf.
    """
    for mid in ("A_base", "B_ft", "C_ft2", "D_quant"):
        db.insert_model(conn, mid, author="test")
    db.add_link(conn, "B_ft", "A_base", "fine_tuned_from")
    db.add_link(conn, "C_ft2", "B_ft", "fine_tuned_from")
    db.add_link(conn, "D_quant", "C_ft2", "quantized_from")
    conn.commit()


def test_pagerank_flows_toward_root(conn):
    """A base model with many derivatives should outrank its leaves. The
    power iteration is the whole reason PageRank works — verifying the
    direction sanity-checks the algorithm without pinning exact values."""
    _seed_lineage_chain(conn)
    scores = pagerank.compute_pagerank(conn)
    assert scores["A_base"] > scores["D_quant"]
    assert scores["A_base"] > scores["C_ft2"]


def test_pagerank_sums_to_one(conn):
    """Standard PageRank invariant: teleport + dangling redistribution
    both preserve probability mass. Off by a hair means dangling mass
    is being lost, and that failure mode is silent otherwise."""
    _seed_lineage_chain(conn)
    scores = pagerank.compute_pagerank(conn)
    total = sum(scores.values())
    assert abs(total - 1.0) < 1e-4


def test_pagerank_empty_graph_returns_empty(conn):
    assert pagerank.compute_pagerank(conn) == {}


def test_pagerank_ignores_non_lineage_relations(conn):
    """`derived_from_dataset` and similar non-lineage edges shouldn't
    influence the score — only the three lineage relations do."""
    db.insert_model(conn, "X", author="test")
    db.insert_model(conn, "Y", author="test")
    db.add_link(conn, "X", "Y", "related_to")  # not a lineage relation
    conn.commit()
    scores = pagerank.compute_pagerank(conn)
    # Both are dangling isolates from PageRank's view — equal rank.
    assert abs(scores.get("X", 0) - scores.get("Y", 0)) < 1e-6


def test_store_pagerank_writes_rows(conn):
    _seed_lineage_chain(conn)
    scores = pagerank.compute_pagerank(conn)
    n = pagerank.store_pagerank(conn, scores)
    assert n == 4
    row = conn.execute(
        "SELECT value FROM model_metadata WHERE model_id='A_base' AND key='pagerank'"
    ).fetchone()
    assert row is not None
    assert float(row["value"]) > 0


def test_inherited_anchors_attenuation(conn):
    """A derivative inherits its parent's anchors at `decay * parent_conf`.
    Depth-2 walk should attenuate further. Missing anchors are ONLY
    written for anchors the derivative doesn't already own."""
    db.insert_model(conn, "parent", author="test")
    db.insert_model(conn, "child", author="test")
    db.add_link(conn, "child", "parent", "fine_tuned_from")
    aid = db.get_or_create_anchor(conn, "parent-only-tag", "CAPABILITY")
    db.link_anchor(conn, "parent", aid, confidence=0.8)
    conn.commit()
    inh = pagerank.inherited_anchors(conn, "child", decay=0.5, max_depth=3)
    assert aid in inh
    # decay=0.5, depth=1, parent_conf=0.8, edge_weight=1.0 → 0.4
    assert abs(inh[aid] - 0.4) < 1e-3


def test_inherited_anchors_skips_own(conn):
    """If the derivative already has the anchor, inheritance is a no-op
    for it — the child's own confidence stands."""
    db.insert_model(conn, "parent", author="test")
    db.insert_model(conn, "child", author="test")
    db.add_link(conn, "child", "parent", "fine_tuned_from")
    aid = db.get_or_create_anchor(conn, "shared-tag", "CAPABILITY")
    db.link_anchor(conn, "parent", aid, confidence=0.8)
    db.link_anchor(conn, "child", aid, confidence=0.6)
    conn.commit()
    inh = pagerank.inherited_anchors(conn, "child", decay=0.5, max_depth=3)
    assert aid not in inh


def test_propagate_lineage_anchors_writes_rows(conn):
    db.insert_model(conn, "P", author="test")
    db.insert_model(conn, "K", author="test")
    db.add_link(conn, "K", "P", "fine_tuned_from")
    aid1 = db.get_or_create_anchor(conn, "cap_x", "CAPABILITY")
    aid2 = db.get_or_create_anchor(conn, "arch_y", "ARCHITECTURE")
    db.link_anchor(conn, "P", aid1, confidence=0.8)
    db.link_anchor(conn, "P", aid2, confidence=0.6)
    conn.commit()
    written = pagerank.propagate_lineage_anchors(conn, decay=0.5, max_depth=3)
    assert written == 2
    kid_anchors = {
        r["anchor_id"]
        for r in conn.execute(
            "SELECT anchor_id FROM model_anchors WHERE model_id='K'"
        ).fetchall()
    }
    assert aid1 in kid_anchors
    assert aid2 in kid_anchors
