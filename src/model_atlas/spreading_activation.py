"""Two-layer spreading activation over the model network.

Sparse-wiki lesson (Collins & Loftus 1975 via `spreading.py`): a signal
placed on one node in a semantic graph propagates two ways —
  * **Layer 1** through *direct edges* between nodes (here: `model_links`)
  * **Layer 2** through *shared anchors* — nodes co-attached to the same
    anchor activate each other even without a direct link between them

Both matter and neither is redundant with the other. A user asking about
`meta-llama/Llama-3.1-8B` should see (via layer 1) its fine-tunes and
quantizations, AND (via layer 2) other 8B-class decoder-only models it
shares anchors with but has no lineage edge to.

Bank-specific channels: each anchor's own bank IS its activation channel.
An anchor in EFFICIENCY contributes to the target's `efficiency`
bank-activation; DOMAIN to `domain`; and so on. The per-bank vector lets
downstream code ask "why is this model related?" with a structured answer
instead of one opaque score.

Not embeddings, not cosine similarity — this is *structural* similarity
via shared discrete anchors weighted by IDF-like sparsity (an anchor
attached to 3 models is a stronger signal than one attached to 3000).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Iterable


@dataclass
class SpreadingConfig:
    """Tunable knobs for `spread()`.

    Defaults are picked for the exploration case (a single-model source,
    want ~50 related). Batch jobs that want deeper reach can raise
    `max_depth` and lower `threshold`; interactive queries that need
    speed can shrink `anchor_limit` (the per-anchor fanout is what
    dominates cost on popular anchors like `transformers-compatible`).
    """

    decay: float = 0.7
    threshold: float = 0.05
    max_depth: int = 2
    max_results: int = 50
    use_anchors: bool = True
    anchor_decay: float = 0.4
    # Popularity cap on layer-2 fanout. `transformers-compatible` covers
    # ~40k models; walking every one for every activation would dominate
    # the whole spread. Higher = richer, slower.
    anchor_limit: int = 12
    # Anchors above this popularity are treated as generic and pruned from
    # layer 2 entirely — they carry near-zero information (every model has
    # them), and their fanout cost is the highest in the graph.
    anchor_popularity_cutoff: int = 5000
    # Per-relation weights on layer 1. `fine_tuned_from` and `quantized_from`
    # are structurally the same edge for spreading purposes, so they share
    # the base weight; `merged_from` is weaker because a merge is a partial
    # inheritance.
    relation_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.relation_weights:
            self.relation_weights = {
                "fine_tuned_from": 1.0,
                "quantized_from": 1.0,
                "merged_from": 0.7,
            }

    def get_weight(self, relation: str) -> float:
        return self.relation_weights.get(relation, 0.5)


@dataclass
class ActivationResult:
    """One activated model plus the trail of how it got there.

    `bank_activations` decomposes the score by anchor bank, so callers can
    say "matches on EFFICIENCY and CAPABILITY but not on ARCHITECTURE"
    instead of a single scalar. `path` and `relations` are for debugging
    / explanation surfaces — a UI that needs to justify "why is this
    model related?" reads them directly.
    """

    model_id: str
    activation: float
    path: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    bank_activations: dict[str, float] = field(default_factory=dict)


def _load_neighbors(
    conn: sqlite3.Connection, model_id: str
) -> list[tuple[str, str, float]]:
    """Layer-1 neighbors: `[(neighbor_id, relation, edge_weight), ...]`.

    Both directions: a fine-tune AND its base model are neighbors of each
    other for spreading purposes. Without that a spread from a base model
    never reaches its own fine-tunes, and vice versa.
    """
    fwd = conn.execute(
        "SELECT target_id, relation, weight FROM model_links WHERE source_id = ?",
        (model_id,),
    ).fetchall()
    rev = conn.execute(
        "SELECT source_id, relation, weight FROM model_links WHERE target_id = ?",
        (model_id,),
    ).fetchall()
    out: list[tuple[str, str, float]] = []
    for tgt, rel, w in fwd:
        out.append((tgt, rel, float(w) if w is not None else 1.0))
    for src, rel, w in rev:
        out.append((src, rel, float(w) if w is not None else 1.0))
    return out


def _load_anchors(
    conn: sqlite3.Connection, model_id: str, max_anchors: int
) -> list[tuple[int, str, str, float]]:
    """This model's anchors: `[(anchor_id, label, bank, model_anchor_weight), ...]`.

    Ordered by the model's own confidence on that anchor so the top-N cut
    keeps its strongest evidence. `max_anchors` bounds fanout even before
    the popularity cutoff kicks in — a heavily-anchored model can otherwise
    dominate the priority queue.
    """
    rows = conn.execute(
        """SELECT ma.anchor_id, a.label, a.bank, ma.confidence
             FROM model_anchors ma
             JOIN anchors a ON a.anchor_id = ma.anchor_id
            WHERE ma.model_id = ?
            ORDER BY ma.confidence DESC
            LIMIT ?""",
        (model_id, max_anchors),
    ).fetchall()
    return [
        (int(r[0]), r[1], r[2], float(r[3]) if r[3] is not None else 1.0) for r in rows
    ]


def _load_anchor_neighbors(
    conn: sqlite3.Connection,
    anchor_id: int,
    limit: int,
    exclude: str,
) -> list[tuple[str, float]]:
    """Models sharing `anchor_id` (excluding `exclude`), highest-confidence first."""
    rows = conn.execute(
        """SELECT model_id, confidence FROM model_anchors
            WHERE anchor_id = ? AND model_id != ?
            ORDER BY confidence DESC
            LIMIT ?""",
        (anchor_id, exclude, limit),
    ).fetchall()
    return [(r[0], float(r[1]) if r[1] is not None else 1.0) for r in rows]


def _anchor_popularity(conn: sqlite3.Connection) -> dict[int, int]:
    """`{anchor_id: model_count}` — one query for the whole cutoff filter."""
    rows = conn.execute(
        "SELECT anchor_id, COUNT(*) FROM model_anchors GROUP BY anchor_id"
    ).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}


def spread(
    conn: sqlite3.Connection,
    sources: Iterable[str] | str,
    *,
    initial_activation: float = 1.0,
    config: SpreadingConfig | None = None,
) -> list[ActivationResult]:
    """Spread activation from one or more source models. Sorted by activation, desc.

    `sources` may be a single `model_id` or an iterable — multi-source
    spreading is the natural query for "models like THESE three", and
    equal-weighting each source is the simplest honest starting condition.
    Sources themselves are excluded from the returned list; the caller
    already knows about them.
    """
    cfg = config or SpreadingConfig()
    if isinstance(sources, str):
        sources = [sources]
    src_ids = list(sources)
    src_set = set(src_ids)

    popularity = _anchor_popularity(conn) if cfg.use_anchors else {}

    # activations: model_id -> (score, path, relations, bank_dict)
    activations: dict[str, tuple[float, list[str], list[str], dict[str, float]]] = {}
    visited: set[str] = set()
    # queue: (-score, depth, model_id, path, relations, bank_dict)
    queue: list[tuple[float, int, str, list[str], list[str], dict[str, float]]] = []

    for sid in src_ids:
        activations[sid] = (initial_activation, [sid], [], {})
        heappush(queue, (-initial_activation, 0, sid, [sid], [], {}))

    while queue and len(visited) < cfg.max_results * 3:
        neg_act, depth, model_id, path, relations, banks = heappop(queue)
        activation = -neg_act
        if model_id in visited:
            continue
        visited.add(model_id)
        if depth >= cfg.max_depth or activation < cfg.threshold:
            continue

        # ── Layer 1: direct model_links ────────────────────────────
        for neighbor_id, relation, edge_w in _load_neighbors(conn, model_id):
            new_act = activation * cfg.decay * cfg.get_weight(relation) * edge_w
            if new_act < cfg.threshold:
                continue
            cur = activations.get(neighbor_id)
            if cur is None or new_act > cur[0]:
                new_path = path + [neighbor_id]
                new_rel = relations + [relation]
                new_banks = dict(banks)
                activations[neighbor_id] = (new_act, new_path, new_rel, new_banks)
                if neighbor_id not in visited:
                    heappush(
                        queue,
                        (-new_act, depth + 1, neighbor_id, new_path, new_rel, new_banks),
                    )

        # ── Layer 2: shared anchors ────────────────────────────────
        if not cfg.use_anchors:
            continue
        for anchor_id, label, bank, ma_w in _load_anchors(conn, model_id, max_anchors=20):
            if popularity.get(anchor_id, 0) > cfg.anchor_popularity_cutoff:
                continue
            for related_id, rel_w in _load_anchor_neighbors(
                conn, anchor_id, cfg.anchor_limit, exclude=model_id
            ):
                anchor_act = activation * cfg.anchor_decay * ma_w * rel_w
                if anchor_act < cfg.threshold:
                    continue
                cur = activations.get(related_id)
                if cur is None or anchor_act > cur[0]:
                    new_path = path + [related_id]
                    new_rel = relations + [f"anchor:{label}"]
                    new_banks = dict(banks)
                    new_banks[bank] = new_banks.get(bank, 0.0) + anchor_act
                    activations[related_id] = (anchor_act, new_path, new_rel, new_banks)
                    if related_id not in visited:
                        heappush(
                            queue,
                            (-anchor_act, depth + 1, related_id, new_path, new_rel, new_banks),
                        )

    results = [
        ActivationResult(
            model_id=mid,
            activation=score,
            path=path,
            relations=relations,
            bank_activations=banks,
        )
        for mid, (score, path, relations, banks) in activations.items()
        if mid not in src_set
    ]
    results.sort(key=lambda r: -r.activation)
    return results[: cfg.max_results]


def anchor_neighbors(
    conn: sqlite3.Connection,
    model_id: str,
    *,
    bank: str | None = None,
    limit: int = 20,
    per_anchor_limit: int = 10,
) -> list[tuple[str, str, float]]:
    """Layer-2-only view: `[(neighbor_id, anchor_label, activation), ...]`.

    Faster than a full `spread(...)` when the caller only wants "what
    shares an anchor with X" and doesn't care about the multi-hop trail.
    `bank` filters to anchors in one bank (EFFICIENCY, ARCHITECTURE, ...);
    None means all banks.
    """
    seen: set[str] = set()
    scored: dict[str, tuple[str, float]] = {}
    for anchor_id, label, anchor_bank, ma_w in _load_anchors(conn, model_id, max_anchors=50):
        if bank and anchor_bank != bank:
            continue
        for related_id, rel_w in _load_anchor_neighbors(
            conn, anchor_id, per_anchor_limit, exclude=model_id
        ):
            if related_id in seen:
                continue
            seen.add(related_id)
            score = ma_w * rel_w
            existing = scored.get(related_id)
            if existing is None or score > existing[1]:
                scored[related_id] = (label, score)
    ranked = sorted(scored.items(), key=lambda kv: -kv[1][1])[:limit]
    return [(mid, label, score) for mid, (label, score) in ranked]


def _bank_activation_summary(
    conn: sqlite3.Connection, results: list[ActivationResult]
) -> dict[str, dict[str, float]]:
    """Aggregate per-model bank activations into `{model_id: {bank: score}}`.

    Convenience for downstream renderers that want a bank-decomposed
    signature per neighbor without walking `ActivationResult.bank_activations`
    themselves. `conn` unused today; retained so the signature can grow to
    normalise against `_anchor_popularity` without a caller update.
    """
    del conn  # reserved for popularity-normalised variants
    return {r.model_id: dict(r.bank_activations) for r in results}


def spread_and_summarize(
    conn: sqlite3.Connection,
    sources: Iterable[str] | str,
    *,
    initial_activation: float = 1.0,
    config: SpreadingConfig | None = None,
) -> tuple[list[ActivationResult], dict[str, dict[str, float]]]:
    """`spread()` + `_bank_activation_summary()` — the shape a query surface uses.

    Returns `(results, {model_id: {bank: score}})`. Results are the ranked
    neighbours; the map is a bank-decomposed reading of each. Two objects,
    one query.
    """
    results = spread(conn, sources, initial_activation=initial_activation, config=config)
    return results, _bank_activation_summary(conn, results)
