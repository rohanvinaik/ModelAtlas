"""Bellman-Ford spreading activation across the semantic network.

Priority-queue spreading adapted from the sparse-wiki-grounding architecture.
Two propagation channels:
  - Link channel: traverse model_links (Layer 1) with relation-specific weights
  - Anchor channel: shared anchors activate co-linked models (Layer 2)

Bank scoping prevents semantic bleeding across unrelated dimensions.
"""

from __future__ import annotations

import heapq
import sqlite3

from .config import (
    ANCHOR_SLICE,
    LINK_WEIGHTS,
    NEIGHBOR_SLICE,
    SPREAD_DECAY,
    SPREAD_MAX_DEPTH,
)


def spread(
    conn: sqlite3.Connection,
    seed_ids: list[str],
    *,
    banks: list[str] | None = None,
    max_depth: int = SPREAD_MAX_DEPTH,
    decay: float = SPREAD_DECAY,
    neighbor_slice: int = NEIGHBOR_SLICE,
    anchor_slice: int = ANCHOR_SLICE,
) -> dict[str, float]:
    """Bellman-Ford spreading activation across the semantic network.

    Returns {model_id: activation_score} for all reached models.

    Args:
        conn: SQLite connection to the network database.
        seed_ids: Model IDs to start spreading from.
        banks: If set, only traverse anchors belonging to these banks.
        max_depth: Maximum hops from any seed.
        decay: Activation multiplier per hop (0.8 = 20% loss per hop).
        neighbor_slice: Max link neighbors explored per node.
        anchor_slice: Max anchor co-occurrences explored per node.
    """
    # activation[model_id] = best activation seen so far
    activation: dict[str, float] = {}
    # Priority queue: (-activation, depth, model_id)
    # Negative activation because heapq is a min-heap
    pq: list[tuple[float, int, str]] = []

    for sid in seed_ids:
        activation[sid] = 1.0
        heapq.heappush(pq, (-1.0, 0, sid))

    while pq:
        neg_act, depth, model_id = heapq.heappop(pq)
        current_act = -neg_act

        # Skip if we've already found a better path
        if current_act < activation.get(model_id, 0.0):
            continue

        if depth >= max_depth:
            continue

        # Channel 1: Link neighbors (Layer 1)
        _spread_links(
            conn,
            model_id,
            current_act,
            depth,
            decay,
            neighbor_slice,
            activation,
            pq,
        )

        # Channel 2: Anchor co-occurrence (Layer 2)
        _spread_anchors(
            conn,
            model_id,
            current_act,
            depth,
            decay,
            anchor_slice,
            banks,
            activation,
            pq,
        )

    return activation


def _spread_links(
    conn: sqlite3.Connection,
    model_id: str,
    current_act: float,
    depth: int,
    decay: float,
    neighbor_slice: int,
    activation: dict[str, float],
    pq: list[tuple[float, int, str]],
) -> None:
    """Propagate activation through explicit model links."""
    # Outgoing links
    rows = conn.execute(
        """SELECT target_id, relation, weight FROM model_links
           WHERE source_id = ? ORDER BY weight DESC LIMIT ?""",
        (model_id, neighbor_slice),
    ).fetchall()

    # Incoming links (bidirectional traversal)
    rows += conn.execute(
        """SELECT source_id, relation, weight FROM model_links
           WHERE target_id = ? ORDER BY weight DESC LIMIT ?""",
        (model_id, neighbor_slice),
    ).fetchall()

    seen: set[str] = set()
    count = 0
    for row in rows:
        neighbor_id = row[0]
        relation = row[1]
        if neighbor_id in seen:
            continue
        seen.add(neighbor_id)

        link_weight = LINK_WEIGHTS.get(relation, 0.5)
        new_act = current_act * decay * link_weight

        if new_act > activation.get(neighbor_id, 0.0):
            activation[neighbor_id] = new_act
            heapq.heappush(pq, (-new_act, depth + 1, neighbor_id))

        count += 1
        if count >= neighbor_slice:
            break


def _spread_anchors(
    conn: sqlite3.Connection,
    model_id: str,
    current_act: float,
    depth: int,
    decay: float,
    anchor_slice: int,
    banks: list[str] | None,
    activation: dict[str, float],
    pq: list[tuple[float, int, str]],
) -> None:
    """Propagate activation through shared anchors (Layer 2)."""
    # Get this model's anchors (optionally bank-scoped)
    if banks:
        placeholders = ",".join("?" for _ in banks)
        anchor_rows = conn.execute(
            f"""SELECT a.anchor_id, a.label FROM model_anchors ma
               JOIN anchors a ON ma.anchor_id = a.anchor_id
               WHERE ma.model_id = ? AND a.bank IN ({placeholders})""",
            (model_id, *banks),
        ).fetchall()
    else:
        anchor_rows = conn.execute(
            """SELECT a.anchor_id, a.label FROM model_anchors ma
               JOIN anchors a ON ma.anchor_id = a.anchor_id
               WHERE ma.model_id = ?""",
            (model_id,),
        ).fetchall()

    if not anchor_rows:
        return

    my_anchor_ids = {r[0] for r in anchor_rows}
    total_anchors = len(my_anchor_ids)

    # Find co-linked models through shared anchors
    placeholders = ",".join("?" for _ in my_anchor_ids)
    co_models = conn.execute(
        f"""SELECT model_id, COUNT(*) as shared
           FROM model_anchors
           WHERE anchor_id IN ({placeholders}) AND model_id != ?
           GROUP BY model_id
           ORDER BY shared DESC
           LIMIT ?""",
        (*my_anchor_ids, model_id, anchor_slice),
    ).fetchall()

    for row in co_models:
        neighbor_id = row[0]
        shared_count = row[1]
        # Weight by fraction of shared anchors
        anchor_weight = shared_count / total_anchors
        new_act = current_act * decay * anchor_weight

        if new_act > activation.get(neighbor_id, 0.0):
            activation[neighbor_id] = new_act
            heapq.heappush(pq, (-new_act, depth + 1, neighbor_id))
