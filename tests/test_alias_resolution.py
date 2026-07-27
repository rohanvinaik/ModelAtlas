"""Tests for anchor-alias resolution on the query path.

The alias table shipped in the corpus and `aliases.py` implemented the
resolver, but nothing ever called it: `require_anchors=["gguf"]` returned zero
results while `["GGUF-available"]` returned matches, despite the alias being
seeded and advertised. These tests pin the wiring, and — more importantly —
pin that resolution can only ever ADD matches: an unresolvable mention must
still constrain the query, never be silently dropped.
"""

from __future__ import annotations

import sqlite3

import pytest

from model_atlas import db
from model_atlas.aliases import add_anchor_alias, canonicalize_labels, resolve_anchor_label
from model_atlas.query_navigate import navigate
from model_atlas.query_types import StructuredQuery


@pytest.fixture
def aliased_conn(conn):
    """A model carrying GGUF-available, reachable by its alias."""
    anchor_id = db.get_or_create_anchor(conn, "GGUF-available", "COMPATIBILITY")
    db.insert_model(conn, "test/quantized-7b", author="test", source="huggingface")
    db.link_anchor(conn, "test/quantized-7b", anchor_id, confidence=1.0)
    db.set_position(conn, "test/quantized-7b", "COMPATIBILITY", 1, 1)
    add_anchor_alias(conn, "gguf", anchor_id, source="test")
    conn.commit()
    return conn


def test_init_db_creates_the_alias_tables(conn):
    """They were only created by `ensure_alias_schema`, which nothing called,
    so a fresh DB had no alias table for the resolver to read."""
    names = {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "anchor_aliases" in names
    assert "model_aliases" in names


def test_alias_resolves_to_canonical_label(aliased_conn):
    assert resolve_anchor_label(aliased_conn, "gguf") == "GGUF-available"


def test_canonical_label_resolves_to_itself(aliased_conn):
    assert resolve_anchor_label(aliased_conn, "GGUF-available") == "GGUF-available"


def test_unknown_mention_resolves_to_nothing(aliased_conn):
    assert resolve_anchor_label(aliased_conn, "not-a-real-anchor") is None


def test_navigate_finds_a_model_by_anchor_alias(aliased_conn):
    """The headline case: `gguf` must find what `GGUF-available` finds."""
    via_alias = navigate(aliased_conn, StructuredQuery(require_anchors=["gguf"]))
    via_canonical = navigate(
        aliased_conn, StructuredQuery(require_anchors=["GGUF-available"])
    )
    assert [r.model_id for r in via_alias] == ["test/quantized-7b"]
    assert [r.model_id for r in via_alias] == [r.model_id for r in via_canonical]


def test_unresolvable_require_anchor_still_returns_nothing(aliased_conn):
    """Resolution must not widen a query by discarding what it cannot read.
    A label naming nothing is still a constraint, and still excludes everything."""
    results = navigate(aliased_conn, StructuredQuery(require_anchors=["invented-label"]))
    assert results == []


def test_canonicalize_reports_unresolved_without_dropping_them(aliased_conn):
    labels, unresolved = canonicalize_labels(aliased_conn, ["gguf", "invented-label"])
    assert labels == ["GGUF-available", "invented-label"]
    assert unresolved == ["invented-label"]


def test_two_spellings_of_one_anchor_collapse(aliased_conn):
    """`gguf` and `GGUF-available` are the same constraint; carrying both
    through would double-count it in the IDF-weighted anchor scoring."""
    labels, unresolved = canonicalize_labels(aliased_conn, ["gguf", "GGUF-available"])
    assert labels == ["GGUF-available"]
    assert unresolved == []


def test_alias_lookup_is_separator_and_case_insensitive(aliased_conn):
    anchor_id = db.get_or_create_anchor(aliased_conn, "mixture-of-experts", "ARCHITECTURE")
    add_anchor_alias(aliased_conn, "moe", anchor_id, source="test")
    aliased_conn.commit()
    for spelling in ("moe", "MoE", "M-o-E", " moe "):
        assert resolve_anchor_label(aliased_conn, spelling) == "mixture-of-experts"


def test_navigate_survives_a_db_without_alias_tables(aliased_conn):
    """A corpus snapshot restored from before the alias tables existed must
    still query — just without aliases."""
    aliased_conn.execute("DROP TABLE anchor_aliases")
    aliased_conn.commit()
    results = navigate(aliased_conn, StructuredQuery(require_anchors=["GGUF-available"]))
    assert [r.model_id for r in results] == ["test/quantized-7b"]


def test_dropped_alias_table_is_reported_as_unknown_nothing(aliased_conn):
    """`_unknown_anchors` must say "cannot tell", not raise, on an old DB."""
    from model_atlas.server import _unknown_anchors

    aliased_conn.execute("DROP TABLE anchor_aliases")
    aliased_conn.commit()
    with pytest.raises(sqlite3.OperationalError):
        canonicalize_labels(aliased_conn, ["gguf"])
    assert _unknown_anchors(aliased_conn, ["gguf"]) == []
