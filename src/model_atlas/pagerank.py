"""PageRank on the lineage graph, and attenuated anchor inheritance for derivatives.

Sparse-wiki lesson: importance is a graph property, not a per-node tag. A model
that many others fine-tune from IS a hub — its own anchor set is stronger
evidence about the neighborhood than a leaf's. PageRank on `model_links`
gives us that importance without curation.

Design choice: inline power iteration instead of a networkx dependency.
The algorithm is 30 lines, deterministic, needs no serialization, and
network.db already ships without heavy graph libs — pulling in networkx
just to compute one score is not a fair trade. If we later want richer
algorithms (community detection, centrality) networkx becomes worth it;
until then this stays inline and auditable.

The second half — attenuated anchor inheritance — is the practical
consequence: a fine-tune of Llama-3-70B has fewer of its own anchors,
but a lot of them are implicit ("it inherits reasoning-tuned, tool-calling,
70B-class from its parent, with the confidence discounted by lineage
distance"). We write those anchors as first-class rows with
`confidence = parent_confidence * decay ** depth` so downstream queries
see them without every caller re-deriving the chain.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

from . import db


# Only relations that carry semantic inheritance. `weight` on the row is
# honored as an edge multiplier (defaults to 1.0), so a `fine_tuned_from`
# with confidence 0.5 pulls less weight than one with 1.0.
_LINEAGE_RELATIONS: tuple[str, ...] = (
    "fine_tuned_from",
    "quantized_from",
    "merged_from",
)


def _load_edges(
    conn: sqlite3.Connection,
    relations: tuple[str, ...] = _LINEAGE_RELATIONS,
) -> tuple[dict[str, list[tuple[str, float]]], set[str]]:
    """Return `(source → [(target, weight), ...], all_nodes)` — only lineage relations.

    The graph is DIRECTED source → target where `source_id fine_tuned_from target_id`
    reads as "source is a fine-tune of target". PageRank flows along these edges to
    the ancestor: the ancestor accumulates importance from its derivatives.
    """
    placeholders = ",".join(["?"] * len(relations))
    rows = conn.execute(
        f"SELECT source_id, target_id, weight FROM model_links WHERE relation IN ({placeholders})",
        relations,
    ).fetchall()
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    nodes: set[str] = set()
    for src, tgt, w in rows:
        out[src].append((tgt, float(w) if w is not None else 1.0))
        nodes.add(src)
        nodes.add(tgt)
    return out, nodes


def compute_pagerank(
    conn: sqlite3.Connection,
    damping: float = 0.85,
    max_iterations: int = 60,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Standard power-iteration PageRank on the lineage graph.

    Dangling nodes (no outgoing lineage — a "root" model like base Llama) have
    their rank redistributed uniformly: without that redistribution their rank
    is stranded and the whole vector fails to sum to 1. Converges when the
    L1 delta drops below `tol`, or after `max_iterations` — measured 30–50
    iterations on 50K nodes.
    """
    edges, nodes = _load_edges(conn)
    if not nodes:
        return {}
    n = len(nodes)
    # Normalize outbound weights per source so a node with 3 lineage parents
    # doesn't push 3× the rank of a node with 1.
    normalized: dict[str, list[tuple[str, float]]] = {}
    for src, targets in edges.items():
        total = sum(w for _, w in targets)
        if total > 0:
            normalized[src] = [(t, w / total) for t, w in targets]
    dangling = [node for node in nodes if node not in normalized]

    rank = {node: 1.0 / n for node in nodes}
    teleport = (1.0 - damping) / n
    for _ in range(max_iterations):
        new_rank = {node: teleport for node in nodes}
        # Dangling nodes redistribute uniformly across the whole node set.
        dangling_mass = damping * sum(rank[d] for d in dangling) / n
        if dangling_mass:
            for node in nodes:
                new_rank[node] += dangling_mass
        for src, targets in normalized.items():
            r = damping * rank[src]
            for tgt, w in targets:
                new_rank[tgt] += r * w
        delta = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if delta < tol:
            break
    return rank


def store_pagerank(conn: sqlite3.Connection, scores: dict[str, float]) -> int:
    """Write `scores` to `model_metadata` under key `pagerank`. Returns rows written.

    Skips scores that would round to zero at 8 decimals — a `1e-9`-scale rank
    for a totally isolated node is noise, not signal, and writing it makes the
    metadata table larger without buying anything.
    """
    n = 0
    for model_id, score in scores.items():
        if score < 1e-8:
            continue
        db.set_metadata(conn, model_id, "pagerank", f"{score:.8f}", "float")
        n += 1
    conn.commit()
    return n


def derive_and_store_pagerank(
    conn: sqlite3.Connection,
    damping: float = 0.85,
    max_iterations: int = 60,
) -> int:
    """One-call convenience: compute + store. Returns rows written."""
    scores = compute_pagerank(conn, damping=damping, max_iterations=max_iterations)
    return store_pagerank(conn, scores)


# ── Attenuated anchor inheritance ─────────────────────────────────────────


def _lineage_parents(
    conn: sqlite3.Connection, model_id: str
) -> list[tuple[str, float]]:
    """Direct lineage parents of `model_id` — the models THIS one derives from.

    Returned as `[(parent_id, edge_weight), ...]`; `edge_weight` is the
    provenance confidence carried on the link row (default 1.0). Multiple
    parents (a merged model) all count.
    """
    placeholders = ",".join(["?"] * len(_LINEAGE_RELATIONS))
    rows = conn.execute(
        f"""SELECT target_id, weight FROM model_links
            WHERE source_id = ? AND relation IN ({placeholders})""",
        (model_id, *_LINEAGE_RELATIONS),
    ).fetchall()
    return [(r[0], float(r[1]) if r[1] is not None else 1.0) for r in rows]


def _model_anchors(
    conn: sqlite3.Connection, model_id: str
) -> dict[int, float]:
    """`{anchor_id: confidence}` — direct anchor links (not inherited)."""
    rows = conn.execute(
        "SELECT anchor_id, confidence FROM model_anchors WHERE model_id = ?",
        (model_id,),
    ).fetchall()
    return {int(r[0]): float(r[1]) for r in rows}


def inherited_anchors(
    conn: sqlite3.Connection,
    model_id: str,
    decay: float = 0.5,
    max_depth: int = 3,
) -> dict[int, float]:
    """Anchors that `model_id` inherits from its lineage ancestors, attenuated.

    Returns `{anchor_id: attenuated_confidence}` where confidence = ancestor's
    own confidence × `decay ** depth` × edge_weight. Only anchors NOT already
    on `model_id` are included — inheritance is fill-in, never overwrite. When
    two ancestor paths reach the same anchor, the higher attenuated confidence
    wins (max, not sum: two witnesses at 0.4 do not add to 0.8 in the honest
    reading).

    Depth-first with `max_depth` cap for the same reason `compute_depth` in
    `hierarchy.py` caps at 32 — a bounded walk is a walk that can be reasoned
    about; unbounded is a walk that can hang on a lineage cycle we introduced
    but never noticed.
    """
    own = set(_model_anchors(conn, model_id).keys())
    inherited: dict[int, float] = {}
    seen: set[str] = {model_id}
    # BFS over lineage: (parent, depth, accumulated_weight_so_far)
    frontier: list[tuple[str, int, float]] = [
        (parent, 1, edge_w) for parent, edge_w in _lineage_parents(conn, model_id)
    ]
    while frontier:
        parent, depth, edge_w = frontier.pop(0)
        if parent in seen or depth > max_depth:
            continue
        seen.add(parent)
        attenuation = (decay ** depth) * edge_w
        for anchor_id, parent_conf in _model_anchors(conn, parent).items():
            if anchor_id in own:
                continue
            attenuated = parent_conf * attenuation
            if attenuated > inherited.get(anchor_id, 0.0):
                inherited[anchor_id] = attenuated
        # Descend
        for grand, next_edge_w in _lineage_parents(conn, parent):
            frontier.append((grand, depth + 1, edge_w * next_edge_w))
    return inherited


def propagate_lineage_anchors(
    conn: sqlite3.Connection,
    decay: float = 0.5,
    max_depth: int = 3,
    min_confidence: float = 0.05,
) -> int:
    """For every model with lineage parents, write attenuated inherited anchors.

    Returns count of anchor rows written. Skips anchors below `min_confidence`
    (0.05 by default) — a 0.001-scale attenuated link is noise-with-provenance;
    the model_anchors table stays legible when we don't record it.

    NOT idempotent-safe as a full re-run: writes use `INSERT OR IGNORE`, so
    a re-run doesn't overwrite (correct — inheritance is fill-in), but a
    manual anchor added after inheritance won't get pushed back to the
    parent. Run this AFTER the corpus is stable, before publishing.
    """
    # Every model that could inherit — i.e., has at least one lineage parent.
    placeholders = ",".join(["?"] * len(_LINEAGE_RELATIONS))
    derivatives = [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT source_id FROM model_links WHERE relation IN ({placeholders})",
            _LINEAGE_RELATIONS,
        ).fetchall()
    ]
    written = 0
    for model_id in derivatives:
        inh = inherited_anchors(conn, model_id, decay=decay, max_depth=max_depth)
        for anchor_id, conf in inh.items():
            if conf < min_confidence:
                continue
            cur = conn.execute(
                """INSERT OR IGNORE INTO model_anchors
                       (model_id, anchor_id, weight, confidence)
                   VALUES (?, ?, ?, ?)""",
                (model_id, anchor_id, conf, conf),
            )
            if cur.rowcount:
                written += 1
    conn.commit()
    return written
