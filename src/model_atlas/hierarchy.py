"""Hierarchical facets — anchors as trees inside each bank.

Sparse-wiki lesson: independent positions on orthogonal *hierarchies* give
additive evidence (depth reduces uncertainty rather than compounding it).
ModelAtlas anchors are today flat within each bank; this module adds the
parent_id column, populates it from declarative parent rules, and exposes
per-anchor depth normalized against the bank's max depth.

Trees seeded:

  EFFICIENCY / size    sub-1B → 1B-class → 3B-class → 7B-class → 13B-class
                                                    → 30B-class → 70B-class
                                                    → frontier-class
  EFFICIENCY / context long-context-32k → 128k → 1m
  LINEAGE   / family    all -family anchors → Base (root)
  DOMAIN                science → physics, chemistry, biology, ...
  ARCHITECTURE          transformer → decoder-only, encoder-only, encoder-decoder
                                    → mixture-of-experts (with decoder-only parent)
                                    → vision-transformer
                        non-transformer roots: mamba, rwkv, ssm, diffusion, hybrid
  COMPATIBILITY / format single root, siblings

Only rows we can assign a parent for get one; unknown-parent stays NULL
(honest abstention, sparse-wiki style — never fabricate a lineage).
"""
from __future__ import annotations

import sqlite3


_HIERARCHY_SCHEMA = """
ALTER TABLE anchors ADD COLUMN parent_id INTEGER DEFAULT NULL REFERENCES anchors(anchor_id);
"""


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def ensure_hierarchy_schema(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "anchors", "parent_id"):
        conn.executescript(_HIERARCHY_SCHEMA)
        conn.commit()


# Parent → children declarative map. Each parent is a label; children are
# labels. Both must exist in BOOTSTRAP_ANCHORS (or be runtime-added).
_HIERARCHY_SEEDS: dict[str, tuple[str, ...]] = {
    # EFFICIENCY size chain — each links to the size below
    "1B-class":       ("sub-1B",),
    "3B-class":       ("1B-class",),
    "7B-class":       ("3B-class",),
    "13B-class":      ("7B-class",),
    "30B-class":      ("13B-class",),
    "70B-class":      ("30B-class",),
    "frontier-class": ("70B-class",),

    # EFFICIENCY context tiers
    "long-context-128k": ("long-context-32k",),
    "long-context-1m":   ("long-context-128k",),

    # ARCHITECTURE — transformer as root, decoder-only + encoder-only + encoder-decoder as children
    "decoder-only":       ("transformer",),
    "encoder-only":       ("transformer",),
    "encoder-decoder":    ("transformer",),
    "mixture-of-experts": ("decoder-only",),   # MoE typically decoder-based
    "vision-transformer": ("encoder-only",),   # ViT is encoder-only style
    "hybrid":             ("transformer",),

    # COMPATIBILITY quantization formats — each specific under generic "quantized"
    # (No parent for base formats — kept flat for now)

    # TRAINING method chain — none seeded, kept flat
}


def _label_id(conn: sqlite3.Connection, label: str) -> int | None:
    row = conn.execute(
        "SELECT anchor_id FROM anchors WHERE label = ?", (label,)
    ).fetchone()
    return int(row[0]) if row else None


def seed_hierarchy(conn: sqlite3.Connection) -> int:
    """Idempotently populate anchors.parent_id from _HIERARCHY_SEEDS.

    Returns the number of parent links written this call."""
    ensure_hierarchy_schema(conn)
    n = 0
    for child, parents in _HIERARCHY_SEEDS.items():
        child_id = _label_id(conn, child)
        if not child_id:
            continue
        # Take the first available parent
        for parent in parents:
            parent_id = _label_id(conn, parent)
            if not parent_id:
                continue
            cur = conn.execute(
                "UPDATE anchors SET parent_id = ? WHERE anchor_id = ? AND parent_id IS NULL",
                (parent_id, child_id),
            )
            if cur.rowcount:
                n += 1
            break
    conn.commit()
    return n


def compute_depth(conn: sqlite3.Connection, anchor_id: int) -> int:
    """Walk from `anchor_id` up to the root, counting hops. Root depth = 0.

    Bounded at 32 hops — a fuse for cycles AND for the type-checker: Pyright
    could not prove the previous `while True` returned an int on every path
    (the cycle-break exited without a return). The `for _ in range(32)` +
    fallthrough `return depth` version is provably bounded and provably
    returns int, without changing behavior — real hierarchies here are 3–4
    deep, so 32 is a cap that only fires under corruption. A cycle exits
    early via the `seen` set; a hit ceiling exits via fallthrough.
    """
    seen: set[int] = set()
    depth = 0
    current = anchor_id
    for _ in range(32):
        if current in seen:
            return depth  # cycle
        seen.add(current)
        row = conn.execute(
            "SELECT parent_id FROM anchors WHERE anchor_id = ?", (current,)
        ).fetchone()
        if not row or row[0] is None:
            return depth
        current = int(row[0])
        depth += 1
    return depth


def bank_max_depth(conn: sqlite3.Connection, bank: str) -> int:
    """Return the maximum depth across all anchors in a bank. Cheap: ~200 anchors
    total, one walk each. Cached per bank in `_bank_depth_cache` if desired."""
    rows = conn.execute(
        "SELECT anchor_id FROM anchors WHERE bank = ?", (bank,)
    ).fetchall()
    if not rows:
        return 0
    return max(compute_depth(conn, int(r[0])) for r in rows)


def normalized_depth(conn: sqlite3.Connection, anchor_id: int) -> float:
    """Return the anchor's depth normalized to [0, 1] against its bank's max.
    Zero-depth roots return 0.0; deepest-in-bank returns 1.0.

    Downstream: navigate_models can use this as an *additive* signal — deeper
    anchors carry stronger evidence about a specific model, shallower ones
    are generic. Matches sparse-wiki's `depth reduces uncertainty` shape.
    """
    row = conn.execute(
        "SELECT bank FROM anchors WHERE anchor_id = ?", (anchor_id,)
    ).fetchone()
    if not row:
        return 0.0
    bank = row[0]
    max_d = bank_max_depth(conn, bank)
    if max_d == 0:
        return 0.0
    my_d = compute_depth(conn, anchor_id)
    return my_d / max_d


def ancestors(conn: sqlite3.Connection, anchor_id: int) -> list[int]:
    """Return anchor_ids on the path from anchor_id (exclusive) to root (inclusive).
    Used by spreading activation to propagate signal to more generic ancestors."""
    out: list[int] = []
    current = anchor_id
    seen: set[int] = {current}
    while True:
        row = conn.execute(
            "SELECT parent_id FROM anchors WHERE anchor_id = ?", (current,)
        ).fetchone()
        if not row or row[0] is None:
            break
        current = int(row[0])
        if current in seen or len(out) > 32:
            break
        seen.add(current)
        out.append(current)
    return out


def descendants(conn: sqlite3.Connection, anchor_id: int) -> list[int]:
    """Return all descendant anchor_ids reachable via parent_id (transitive).
    Cheap — one recursive CTE."""
    rows = conn.execute(
        """WITH RECURSIVE tree(id) AS (
             SELECT anchor_id FROM anchors WHERE parent_id = ?
             UNION ALL
             SELECT a.anchor_id FROM anchors a JOIN tree ON a.parent_id = tree.id
           )
           SELECT id FROM tree""",
        (anchor_id,),
    ).fetchall()
    return [int(r[0]) for r in rows]
