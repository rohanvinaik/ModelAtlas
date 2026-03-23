"""Tests for spreading.py internal helpers — _spread_links and _spread_anchors.

These void functions mutate shared state (activation dict + priority queue).
Tests verify effects through the mutated state.
"""

import sqlite3

import pytest

from model_atlas import db
from model_atlas.spreading import _spread_anchors, _spread_links


@pytest.fixture
def spread_db(tmp_path):
    """Create a test DB with models, links, and anchors for spreading tests."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    db.init_db(conn)

    # Insert models
    for mid in ["A", "B", "C", "D"]:
        db.insert_model(conn, mid)

    # Insert links: A -> B (fine_tuned_from), A -> C (same_family)
    db.add_link(conn, "A", "B", "fine_tuned_from", 0.9)
    db.add_link(conn, "A", "C", "same_family", 0.7)

    # Insert anchors and link them
    aid1 = db.get_or_create_anchor(conn, "tool-calling", "CAPABILITY")
    aid2 = db.get_or_create_anchor(conn, "code-domain", "DOMAIN")
    aid3 = db.get_or_create_anchor(conn, "instruction-following", "CAPABILITY")

    # A has tool-calling + code-domain
    db.link_anchor(conn, "A", aid1)
    db.link_anchor(conn, "A", aid2)
    # B has tool-calling + instruction-following
    db.link_anchor(conn, "B", aid1)
    db.link_anchor(conn, "B", aid3)
    # D has code-domain (shares with A)
    db.link_anchor(conn, "D", aid2)

    conn.commit()
    return conn


class TestSpreadLinks:
    """Test _spread_links by observing mutations to activation dict and pq."""

    def test_outgoing_links_activate_neighbors(self, spread_db):
        activation = {"A": 1.0}
        pq: list[tuple[float, int, str]] = []
        _spread_links(spread_db, "A", 1.0, 0, 0.8, 10, activation, pq)
        # B should be activated via fine_tuned_from (weight 0.9)
        assert "B" in activation
        assert activation["B"] == pytest.approx(1.0 * 0.8 * 0.9)

    def test_incoming_links_bidirectional(self, spread_db):
        activation = {"B": 1.0}
        pq: list[tuple[float, int, str]] = []
        _spread_links(spread_db, "B", 1.0, 0, 0.8, 10, activation, pq)
        # A should be activated via incoming link (B is target of A->B)
        assert "A" in activation

    def test_decay_applied(self, spread_db):
        activation = {"A": 1.0}
        pq: list[tuple[float, int, str]] = []
        _spread_links(spread_db, "A", 1.0, 0, 0.5, 10, activation, pq)
        # With decay=0.5 and link weight 0.9: 1.0 * 0.5 * 0.9 = 0.45
        assert activation.get("B", 0) == pytest.approx(0.45)

    def test_neighbor_slice_limits(self, spread_db):
        activation = {"A": 1.0}
        pq: list[tuple[float, int, str]] = []
        _spread_links(spread_db, "A", 1.0, 0, 0.8, 1, activation, pq)
        # Only 1 neighbor allowed — should get B or C but not both
        activated = [k for k in activation if k != "A"]
        assert len(activated) <= 1

    def test_pq_populated(self, spread_db):
        activation = {"A": 1.0}
        pq: list[tuple[float, int, str]] = []
        _spread_links(spread_db, "A", 1.0, 0, 0.8, 10, activation, pq)
        # Priority queue should have entries for activated neighbors
        assert len(pq) > 0
        # PQ entries are (-activation, depth+1, model_id)
        neg_act, depth, mid = pq[0]
        assert neg_act < 0
        assert depth == 1

    def test_no_update_if_existing_better(self, spread_db):
        activation = {"A": 1.0, "B": 0.99}  # B already has high activation
        pq: list[tuple[float, int, str]] = []
        _spread_links(spread_db, "A", 1.0, 0, 0.8, 10, activation, pq)
        # New activation for B = 1.0 * 0.8 * 0.9 = 0.72 < 0.99
        assert activation["B"] == 0.99  # unchanged

    def test_dedup_seen_set(self, spread_db):
        """Same neighbor appearing in both outgoing and incoming shouldn't double-activate."""
        activation = {"A": 1.0}
        pq: list[tuple[float, int, str]] = []
        _spread_links(spread_db, "A", 1.0, 0, 0.8, 10, activation, pq)
        # Count PQ entries for B — should be exactly 1
        b_entries = [e for e in pq if e[2] == "B"]
        assert len(b_entries) == 1


class TestSpreadAnchors:
    """Test _spread_anchors by observing mutations to activation dict."""

    def test_shared_anchors_activate(self, spread_db):
        activation = {"A": 1.0}
        pq: list[tuple[float, int, str]] = []
        _spread_anchors(spread_db, "A", 1.0, 0, 0.8, 10, None, activation, pq)
        # D shares code-domain with A, should be activated
        assert "D" in activation

    def test_bank_scoping(self, spread_db):
        activation = {"A": 1.0}
        pq: list[tuple[float, int, str]] = []
        _spread_anchors(spread_db, "A", 1.0, 0, 0.8, 10, ["DOMAIN"], activation, pq)
        # Only DOMAIN anchors considered — D shares code-domain
        assert "D" in activation

    def test_bank_scoping_excludes(self, spread_db):
        activation = {"A": 1.0}
        pq: list[tuple[float, int, str]] = []
        # Scope to a bank A has no anchors in
        _spread_anchors(spread_db, "A", 1.0, 0, 0.8, 10, ["TRAINING"], activation, pq)
        # No anchors in TRAINING bank → no spreading
        assert "D" not in activation

    def test_anchor_slice_limits(self, spread_db):
        activation = {"A": 1.0}
        pq: list[tuple[float, int, str]] = []
        _spread_anchors(spread_db, "A", 1.0, 0, 0.8, 1, None, activation, pq)
        # Limited to 1 anchor neighbor
        activated = [k for k in activation if k != "A"]
        assert len(activated) <= 1

    def test_no_activation_for_model_without_anchors(self, spread_db):
        activation = {"C": 1.0}  # C has no anchors
        pq: list[tuple[float, int, str]] = []
        _spread_anchors(spread_db, "C", 1.0, 0, 0.8, 10, None, activation, pq)
        # No anchors → no spreading → only C in activation
        assert len([k for k in activation if k != "C"]) == 0

    def test_decay_and_fraction(self, spread_db):
        activation = {"A": 1.0}
        pq: list[tuple[float, int, str]] = []
        _spread_anchors(spread_db, "A", 1.0, 0, 0.8, 10, None, activation, pq)
        # B shares 1 anchor with A (tool-calling), A has 2 anchors total
        # fraction = 1/2 = 0.5, new_act = 1.0 * 0.8 * 0.5 = 0.4
        if "B" in activation:
            assert activation["B"] == pytest.approx(0.4)
