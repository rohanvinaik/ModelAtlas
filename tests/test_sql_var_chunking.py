"""Tests for chunking candidate sets under SQLite's bind-parameter cap.

Regression guard. SQLite caps bind parameters per statement
(SQLITE_MAX_VARIABLE_NUMBER: 32766 since 3.32, 999 on older builds). A query
with no `require_anchors` makes the whole corpus the candidate set, so every
model id was splatted into one `IN (...)` — which raised
`OperationalError: too many SQL variables` on the real 50,906-model corpus
while passing on every small fixture in the suite. These tests build a set
larger than the cap so the boundary is exercised, not assumed.
"""

from __future__ import annotations

import sqlite3

import pytest

from model_atlas import db
from model_atlas.db_queries import SQL_VAR_CHUNK, chunked
from model_atlas.query_navigate import _anchor_counts_over

# One model past the modern cap — enough to fail without chunking, small
# enough to build in well under a second.
OVER_CAP = 32_767


def test_chunks_are_disjoint_and_cover_exactly_once():
    """The count-summing callers depend on this: a model must land in exactly
    one chunk, or per-chunk COUNT(DISTINCT ...) sums would double-count."""
    items = [f"m{i}" for i in range(2_500)]
    chunks = list(chunked(items, size=900))
    assert [c for chunk in chunks for c in chunk] == items
    assert sum(len(c) for c in chunks) == len(items)
    assert all(len(c) <= 900 for c in chunks)


def test_chunk_size_is_under_the_oldest_sqlite_floor():
    """999 was the pre-3.32 default. Staying under it keeps the same code
    correct on an old SQLite, not just the one this machine happens to ship."""
    assert SQL_VAR_CHUNK < 999


def test_empty_input_yields_no_chunks():
    assert list(chunked([])) == []


@pytest.fixture
def big_conn(conn):
    """A corpus larger than SQLite's bind-parameter cap."""
    ids = [f"a/m{i}" for i in range(OVER_CAP)]
    conn.executemany(
        "INSERT INTO models (model_id, author, source) VALUES (?, 'a', 'test')",
        [(i,) for i in ids],
    )
    conn.executemany(
        "INSERT INTO model_positions (model_id, bank, path_sign, path_depth) "
        "VALUES (?, 'EFFICIENCY', -1, 1)",
        [(i,) for i in ids],
    )
    anchor_id = db.get_or_create_anchor(conn, "chat", "CAPABILITY")
    conn.executemany(
        "INSERT INTO model_anchors (model_id, anchor_id) VALUES (?, ?)",
        [(i, anchor_id) for i in ids],
    )
    conn.commit()
    return conn, ids


def test_raw_in_clause_over_the_cap_really_does_raise(big_conn):
    """Pins the failure mode itself. If SQLite ever stops raising here, the
    tests below stop proving anything and should be revisited."""
    conn, ids = big_conn
    ph = ",".join("?" for _ in ids)
    with pytest.raises(sqlite3.OperationalError, match="too many SQL variables"):
        conn.execute(f"SELECT model_id FROM models WHERE model_id IN ({ph})", ids)


def test_batch_get_positions_over_the_cap(big_conn):
    conn, ids = big_conn
    out = db.batch_get_positions(conn, ids)
    assert len(out) == OVER_CAP
    assert out[ids[0]]["EFFICIENCY"] == (-1, 1)


def test_batch_get_anchor_sets_over_the_cap(big_conn):
    conn, ids = big_conn
    out = db.batch_get_anchor_sets(conn, ids)
    assert len(out) == OVER_CAP
    assert out[ids[-1]] == {"chat"}


def test_anchor_counts_sum_correctly_across_chunks(big_conn):
    """The aggregating path: per-chunk COUNT(DISTINCT model_id) summed must
    equal what one unchunked GROUP BY would have returned."""
    conn, ids = big_conn
    counts = _anchor_counts_over(conn, ids)
    assert counts["chat"] == OVER_CAP


def test_navigate_over_the_cap_with_no_require_anchors(big_conn):
    """The end-to-end reproduction: no `require_anchors` → the whole corpus
    is the candidate set → every batch helper sees more ids than the cap."""
    from model_atlas.query_navigate import navigate
    from model_atlas.query_types import StructuredQuery

    conn, _ = big_conn
    results = navigate(conn, StructuredQuery(efficiency=-1, limit=5))
    assert len(results) == 5
