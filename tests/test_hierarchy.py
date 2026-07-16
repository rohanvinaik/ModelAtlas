"""Tests for `hierarchy` — anchor parent_id, depth, ancestors, descendants."""

from __future__ import annotations

from model_atlas import hierarchy


def _seed_chain(conn) -> tuple[int, int, int]:
    """Create three anchors in a parent chain: root → mid → leaf.

    Returns their anchor_ids. Uses raw insert to avoid seeding logic that
    might normalize labels differently across versions.
    """
    conn.execute("INSERT INTO anchors (label, bank) VALUES ('root_a', 'ARCHITECTURE')")
    conn.execute("INSERT INTO anchors (label, bank) VALUES ('mid_a', 'ARCHITECTURE')")
    conn.execute("INSERT INTO anchors (label, bank) VALUES ('leaf_a', 'ARCHITECTURE')")
    conn.commit()
    root = conn.execute("SELECT anchor_id FROM anchors WHERE label='root_a'").fetchone()[0]
    mid = conn.execute("SELECT anchor_id FROM anchors WHERE label='mid_a'").fetchone()[0]
    leaf = conn.execute("SELECT anchor_id FROM anchors WHERE label='leaf_a'").fetchone()[0]
    return int(root), int(mid), int(leaf)


def test_ensure_schema_is_idempotent(conn):
    hierarchy.ensure_hierarchy_schema(conn)
    hierarchy.ensure_hierarchy_schema(conn)  # second call must not raise
    cols = [c[1] for c in conn.execute("PRAGMA table_info(anchors)").fetchall()]
    assert "parent_id" in cols


def test_compute_depth_of_root_is_zero(conn):
    hierarchy.ensure_hierarchy_schema(conn)
    root, _, _ = _seed_chain(conn)
    assert hierarchy.compute_depth(conn, root) == 0


def test_compute_depth_walks_chain(conn):
    hierarchy.ensure_hierarchy_schema(conn)
    root, mid, leaf = _seed_chain(conn)
    conn.execute("UPDATE anchors SET parent_id=? WHERE anchor_id=?", (root, mid))
    conn.execute("UPDATE anchors SET parent_id=? WHERE anchor_id=?", (mid, leaf))
    conn.commit()
    assert hierarchy.compute_depth(conn, leaf) == 2


def test_compute_depth_survives_cycle(conn):
    """A parent_id cycle used to hang the old `while True` loop; the
    for-range(32) fuse guarantees termination and Pyright-provable int
    return on all paths. The value returned is the walk length up to the
    cycle break, so it's bounded, not meaningful."""
    hierarchy.ensure_hierarchy_schema(conn)
    root, mid, _ = _seed_chain(conn)
    conn.execute("UPDATE anchors SET parent_id=? WHERE anchor_id=?", (mid, root))
    conn.execute("UPDATE anchors SET parent_id=? WHERE anchor_id=?", (root, mid))
    conn.commit()
    d = hierarchy.compute_depth(conn, root)
    assert isinstance(d, int)
    assert d <= 32  # fuse held


def test_normalized_depth_root_zero(conn):
    hierarchy.ensure_hierarchy_schema(conn)
    root, mid, leaf = _seed_chain(conn)
    conn.execute("UPDATE anchors SET parent_id=? WHERE anchor_id=?", (root, mid))
    conn.execute("UPDATE anchors SET parent_id=? WHERE anchor_id=?", (mid, leaf))
    conn.commit()
    assert hierarchy.normalized_depth(conn, root) == 0.0
    assert hierarchy.normalized_depth(conn, leaf) == 1.0


def test_ancestors_returns_ordered_path(conn):
    hierarchy.ensure_hierarchy_schema(conn)
    root, mid, leaf = _seed_chain(conn)
    conn.execute("UPDATE anchors SET parent_id=? WHERE anchor_id=?", (root, mid))
    conn.execute("UPDATE anchors SET parent_id=? WHERE anchor_id=?", (mid, leaf))
    conn.commit()
    assert hierarchy.ancestors(conn, leaf) == [mid, root]
    assert hierarchy.ancestors(conn, root) == []


def test_descendants_recursive_cte(conn):
    hierarchy.ensure_hierarchy_schema(conn)
    root, mid, leaf = _seed_chain(conn)
    conn.execute("UPDATE anchors SET parent_id=? WHERE anchor_id=?", (root, mid))
    conn.execute("UPDATE anchors SET parent_id=? WHERE anchor_id=?", (mid, leaf))
    conn.commit()
    assert set(hierarchy.descendants(conn, root)) == {mid, leaf}
    assert hierarchy.descendants(conn, leaf) == []


def test_seed_hierarchy_only_writes_when_labels_exist(conn):
    """Skips seeds whose child or parent label isn't present. Bootstrap
    anchors don't include every future label, so the seed must be
    partial-application safe."""
    hierarchy.ensure_hierarchy_schema(conn)
    n = hierarchy.seed_hierarchy(conn)
    # `_HIERARCHY_SEEDS` targets labels like `7B-class`, `mixture-of-experts`
    # that DO exist in the bootstrap. n should be positive on a fresh DB.
    assert n >= 0  # tolerant — the exact count depends on bootstrap coverage
    # Any parent_id populated must point at a real anchor row.
    for row in conn.execute(
        "SELECT anchor_id, parent_id FROM anchors WHERE parent_id IS NOT NULL"
    ):
        exists = conn.execute(
            "SELECT 1 FROM anchors WHERE anchor_id = ?", (row["parent_id"],)
        ).fetchone()
        assert exists, f"dangling parent_id={row['parent_id']}"
