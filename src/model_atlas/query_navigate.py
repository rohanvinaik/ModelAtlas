"""Extracted from src/model_atlas/query.py."""

from __future__ import annotations

import sqlite3

from . import db
from .config import (
    NAVIGATE_AVOID_DECAY,
    NAVIGATE_MISSING_BANK_PENALTY,
)
from .query_types import (  # noqa: F401 — re-exported for backward compat
    BankConstraint,
    ComparisonResult,
    NavigationResult,
    ParsedQuery,
    SearchResult,
    StructuredQuery,
)

# Module-level IDF cache, invalidated by clearing it after index builds.
_idf_cache: dict[str, float] = {}


def _get_idf(conn: sqlite3.Connection) -> dict[str, float]:
    """Return cached IDF dict, computing on first call."""
    global _idf_cache
    if not _idf_cache:
        _idf_cache = db.compute_anchor_idf(conn)
    return _idf_cache


def invalidate_idf_cache() -> None:
    """Clear the IDF cache — call after index builds."""
    global _idf_cache
    _idf_cache = {}


def _bank_score_single(model_signed_pos: int, query_direction: int) -> float:
    """Score a single bank alignment.

    direction == 0: want near zero, penalize distance.
    direction == +1/-1: reward alignment, penalize opposition.
    """
    if query_direction == 0:
        return 1.0 / (1.0 + abs(model_signed_pos))
    alignment = model_signed_pos * query_direction
    if alignment > 0:
        return 1.0
    elif alignment == 0:
        return 0.5
    else:
        return 1.0 / (1.0 + abs(alignment))


def _nav_candidates(
    conn: sqlite3.Connection,
    require_set: set[str],
) -> list[str] | None:
    """Resolve candidate model IDs, pre-filtering by required anchors.

    Returns None if a required anchor doesn't exist (→ no results possible).
    """
    if require_set:
        anchor_ids = []
        for label in require_set:
            row = conn.execute(
                "SELECT anchor_id FROM anchors WHERE label = ?", (label,)
            ).fetchone()
            if row:
                anchor_ids.append(row["anchor_id"])
            else:
                return None
        if len(anchor_ids) != len(require_set):
            return None
        placeholders = ",".join("?" for _ in anchor_ids)
        rows = conn.execute(
            f"""SELECT model_id FROM model_anchors
                WHERE anchor_id IN ({placeholders})
                GROUP BY model_id
                HAVING COUNT(DISTINCT anchor_id) = ?""",
            [*anchor_ids, len(anchor_ids)],
        ).fetchall()
        return [r["model_id"] for r in rows] or None
    rows = conn.execute("SELECT model_id FROM models").fetchall()
    return [r["model_id"] for r in rows] or None


def _nav_bank_alignment(
    positions: dict[str, tuple[int, int]],
    directions: dict[str, int],
) -> float:
    """Multiplicative bank alignment score across all queried banks."""
    if not directions:
        return 1.0
    result = 1.0
    for bank_name, direction in directions.items():
        pos = positions.get(bank_name)
        if pos is None:
            result *= NAVIGATE_MISSING_BANK_PENALTY
        else:
            result *= _bank_score_single(pos[0] * pos[1], direction)
    return result


def _nav_anchor_relevance(
    model_anchor_set: set[str],
    prefer_set: set[str],
    avoid_set: set[str],
    idf: dict[str, float],
    prefer_idf_total: float,
    has_constraints: bool,
) -> float:
    """IDF-weighted anchor relevance combining prefer/avoid signals."""
    if not has_constraints:
        return 1.0
    if prefer_set and prefer_idf_total > 0:
        matched = prefer_set & model_anchor_set
        prefer_score = sum(idf.get(a, 0.0) for a in matched) / prefer_idf_total
    else:
        prefer_score = 1.0
    avoided = avoid_set & model_anchor_set
    avoid_penalty = NAVIGATE_AVOID_DECAY ** len(avoided)
    return prefer_score * avoid_penalty



def _nav_context_bias(
    model_anchor_set: set[str],
    context_set: set[str],
    idf: dict[str, float],
) -> float:
    """Multiplicative bias in `[1.0, 1 + max_boost]` from context-anchor intersection.

    The IDF-weighted fraction of `context_set` this model already carries is
    added on top of 1.0 — so a model matching the whole context gets a modest
    boost (up to ~1.5×), while a model matching zero context gets 1.0
    (neutral, not penalised). Absent context is neutral — the field is
    additive information, never a filter. Sparse-wiki's "same mention,
    different context" applied at scoring time: `jordan` next to
    `basketball` biases toward the basketball-player anchor set without
    excluding country / river / professor from the ranking.
    """
    if not context_set:
        return 1.0
    matched = context_set & model_anchor_set
    if not matched:
        return 1.0
    ctx_idf_total = sum(idf.get(a, 0.0) for a in context_set)
    if ctx_idf_total <= 0:
        return 1.0
    matched_idf = sum(idf.get(a, 0.0) for a in matched)
    # Cap the boost at 1.5× — a soft bias, not a hard override; keeps
    # context from steamrolling the rest of the constraint stack when a
    # caller happens to pass a well-covered context set.
    return 1.0 + 0.5 * (matched_idf / ctx_idf_total)



NAVIGATE_PAGERANK_MAX_BOOST = 0.2
"""Cap on the PageRank multiplier. `1 + this` is what a top-PR candidate can
gain over a zero-PR one — kept small so PageRank is a soft prior (a nudge
between two nearly-tied candidates), never enough to invert a bank/anchor
mismatch. 0.2 = up to 1.2×, matching the coherence/context bias scale so no
single soft signal steamrolls the others."""


def _nav_pagerank_boost(model_pr: float, pr_max: float) -> float:
    """Multiplicative boost from PageRank, normalized against `pr_max`.

    Returns 1.0 when `pr_max` is zero (no candidate has PageRank) or when
    the candidate has no stored score — absent is neutral. Otherwise
    `1 + MAX_BOOST * (pr / pr_max)`, so the top-PR candidate in the set
    gets exactly `1 + MAX_BOOST` and everything below scales linearly.
    """
    if pr_max <= 0 or model_pr <= 0:
        return 1.0
    return 1.0 + NAVIGATE_PAGERANK_MAX_BOOST * (model_pr / pr_max)


NAVIGATE_EPA_HALF_LIFE = 0.6
"""Distance in EPA-space at which similarity drops to 0.5. Chosen so a
candidate 0.6 away on ONE requested axis reads as "notably off" (0.5),
and 1.2 away reads as "clearly wrong" (0.25). Full EPA range on one axis
is 2 (±1), so this puts the 50% mark at ~30% of the range — a
recognizable qualitative gap."""


def _nav_epa_alignment(
    model_epa: tuple[float, float, float] | None,
    target: tuple[float | None, float | None, float | None],
) -> float:
    """Similarity between candidate EPA and target axes. Returns [0, 1].

    Uses per-axis absolute distance with a soft half-life decay. Only the
    axes the caller specified count — an unrestricted axis contributes
    nothing (positive or negative) to the score. Candidate with no stored
    EPA returns 1.0 (neutral) — abstention is symmetric: the tool doesn't
    penalize a model for lacking a vibe_summary any more than it rewards
    a false-neutral (0, 0, 0) reading.
    """
    te, tp, ta = target
    specified = [(t, i) for i, t in enumerate((te, tp, ta)) if t is not None]
    if not specified:
        return 1.0
    if model_epa is None:
        return 1.0  # abstention is symmetric — see docstring
    # Half-life decay: score = 0.5 ** (|dist| / half_life), product across axes
    score = 1.0
    for target_val, axis_idx in specified:
        dist = abs(model_epa[axis_idx] - target_val)
        score *= 0.5 ** (dist / NAVIGATE_EPA_HALF_LIFE)
    return score


def _nav_seed_similarity(
    model_anchor_set: set[str],
    seed_anchors: set[str],
    idf: dict[str, float],
) -> float:
    """IDF-weighted Jaccard between seed model anchors and candidate."""
    if not seed_anchors:
        return 1.0
    shared = seed_anchors & model_anchor_set
    union = seed_anchors | model_anchor_set
    idf_union = sum(idf.get(a, 0.0) for a in union)
    if not idf_union:
        return 0.0
    return sum(idf.get(a, 0.0) for a in shared) / idf_union


def _mark_tie_clusters(
    results: list[NavigationResult],
    epsilon: float = 0.05,
    max_cluster_size: int = 20,
) -> None:
    """Mutate `results` in place: mark contiguous score-runs within `epsilon` as
    tie-clusters, and stamp each cluster member with the discriminating axis.

    Discriminating axis = the position bank with the highest cross-member
    variance in `sign * (1 + depth)`. If a bank splits the cluster (e.g. GGUF
    vs safetensors on COMPATIBILITY), variance is high; if every member shares
    the same sign/depth for that bank, variance is zero. Ties on max variance
    are broken by bank name for determinism. Sparse-wiki's fail-honest
    ordering — the tool cannot invent a tiebreaker the anchors don't already
    encode, so it names the axis that would earn one.

    `max_cluster_size` caps a run because a very long tie-cluster (100
    results all at score 1.0) has no useful "which axis breaks this" answer;
    the top-N is what the caller sees, so only mark within the first N.
    """
    if not results:
        return
    n = len(results)
    cluster_id = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and j - i < max_cluster_size and results[j - 1].score - results[j].score < epsilon:
            j += 1
        if j - i >= 2:
            axis = _discriminating_axis([results[k].positions for k in range(i, j)])
            for k in range(i, j):
                results[k].tie_cluster_id = cluster_id
                results[k].discriminating_axis = axis
            cluster_id += 1
        i = j


def _discriminating_axis(positions_list: list[dict[str, dict]]) -> str | None:
    """Bank with highest variance in `sign * (1 + depth)` across members.

    None when no bank varies (every member positions identically — a real
    "we cannot break this" answer). A bank missing on a member counts as
    (0, 0) — an absence IS variance from a present position; without that,
    "some members have COMPATIBILITY, some don't" would silently score zero
    and get missed as a discriminator.
    """
    all_banks: set[str] = set()
    for p in positions_list:
        all_banks.update(p.keys())
    best_bank: str | None = None
    best_var = 0.0
    for bank in sorted(all_banks):  # sorted → deterministic tie-break
        vals = []
        for p in positions_list:
            pos = p.get(bank)
            if pos is None:
                vals.append(0.0)
            else:
                sign = pos.get("sign", 0)
                depth = pos.get("depth", 0)
                vals.append(float(sign) * (1.0 + float(depth)))
        # Variance without stdlib: (mean of squares) - (mean)^2
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        if var > best_var:
            best_var = var
            best_bank = bank
    return best_bank


def navigate(
    conn: sqlite3.Connection,
    query: StructuredQuery,
) -> list[NavigationResult]:
    """Structured navigational search — the primary recommendation engine.

    Four signals, multiplicative:
        final_score = bank_alignment * anchor_relevance * seed_similarity * coherence

    The coherence factor comes from the certifier's per-model
    certification_score (Phase 6 of the audit pipeline). It defaults to 1.0
    when unpopulated so pre-recert data continues to work.

    Uses batch SQL instead of N+1 per-model lookups.
    """
    idf = _get_idf(conn)
    directions = query.bank_directions()
    require_set = set(query.require_anchors)
    prefer_set = set(query.prefer_anchors)
    avoid_set = set(query.avoid_anchors)
    context_set = set(query.context_anchors)
    has_anchor_constraints = bool(require_set or prefer_set or avoid_set)
    prefer_idf_total = sum(idf.get(a, 0.0) for a in prefer_set) if prefer_set else 0.0

    # Step 1: Determine candidate set
    candidate_ids = _nav_candidates(conn, require_set)
    if not candidate_ids:
        return []

    # Step 2: Batch-fetch positions, anchors, authors
    all_positions = db.batch_get_positions(conn, candidate_ids)
    all_anchors = db.batch_get_anchor_sets(conn, candidate_ids)
    seed_anchors = (
        db.get_anchor_set(conn, query.similar_to) if query.similar_to else set()
    )

    ph = ",".join("?" for _ in candidate_ids)
    author_rows = conn.execute(
        f"SELECT model_id, author FROM models WHERE model_id IN ({ph})",
        candidate_ids,
    ).fetchall()
    authors = {r["model_id"]: r["author"] or "" for r in author_rows}

    # Batch-fetch certification scores (Phase 6 coherence). Missing = 1.0.
    # Also pull pagerank + EPA in the SAME query — one round-trip for every
    # per-model soft signal navigate consumes. Adding another metadata read
    # per key would N+1 the score loop; a single IN + WHERE key IN (...)
    # stays O(1) queries regardless of how many soft signals we wire in.
    meta_rows = conn.execute(
        f"""SELECT model_id, key, value FROM model_metadata
            WHERE key IN ('certification_score', 'pagerank', 'vibe_e', 'vibe_p', 'vibe_a')
              AND model_id IN ({ph})""",
        candidate_ids,
    ).fetchall()
    coherence_by_id: dict[str, float] = {}
    pagerank_by_id: dict[str, float] = {}
    epa_by_id: dict[str, tuple[float, float, float]] = {}
    _epa_tmp: dict[str, dict[str, float]] = {}
    for row in meta_rows:
        try:
            v = float(row["value"])
        except (TypeError, ValueError):
            continue
        key = row["key"]
        mid = row["model_id"]
        if key == "certification_score":
            coherence_by_id[mid] = v
        elif key == "pagerank":
            pagerank_by_id[mid] = v
        elif key in ("vibe_e", "vibe_p", "vibe_a"):
            _epa_tmp.setdefault(mid, {})[key] = v
    for mid, d in _epa_tmp.items():
        if len(d) == 3:  # all-or-nothing, matches vibe_axes.load_epa
            epa_by_id[mid] = (d["vibe_e"], d["vibe_p"], d["vibe_a"])
    # Normalize pagerank against the max IN THE CANDIDATE SET (not the whole
    # corpus) — a boost of "top of these results" reads more naturally than
    # "top of the entire corpus" for a filtered query. Multiplicative factor
    # in [1.0, 1 + PAGERANK_MAX_BOOST].
    _pr_max = max(pagerank_by_id.values(), default=0.0)

    # Step 3: Score each candidate
    results: list[NavigationResult] = []
    for mid in candidate_ids:
        positions = all_positions.get(mid, {})
        model_anchor_set = all_anchors.get(mid, set())

        bank_alignment = _nav_bank_alignment(positions, directions)
        anchor_relevance = _nav_anchor_relevance(
            model_anchor_set,
            prefer_set,
            avoid_set,
            idf,
            prefer_idf_total,
            has_anchor_constraints,
        )
        seed_similarity = _nav_seed_similarity(model_anchor_set, seed_anchors, idf)
        coherence = coherence_by_id.get(mid, 1.0)
        context_bias = _nav_context_bias(model_anchor_set, context_set, idf)
        pagerank_boost = _nav_pagerank_boost(pagerank_by_id.get(mid, 0.0), _pr_max)
        epa_alignment = _nav_epa_alignment(
            epa_by_id.get(mid), (query.vibe_e, query.vibe_p, query.vibe_a)
        )
        final_score = (
            bank_alignment
            * anchor_relevance
            * seed_similarity
            * coherence
            * context_bias
            * pagerank_boost
            * epa_alignment
        )

        pos_out = {
            bank_name: {"sign": sign, "depth": depth}
            for bank_name, (sign, depth) in positions.items()
        }
        results.append(
            NavigationResult(
                model_id=mid,
                score=final_score,
                bank_alignment=bank_alignment,
                anchor_relevance=anchor_relevance,
                seed_similarity=seed_similarity,
                coherence=coherence,
                positions=pos_out,
                anchor_labels=sorted(model_anchor_set),
                author=authors.get(mid, ""),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    top = results[: query.limit]
    # Cluster BEFORE truncating further — a cluster that straddles the limit
    # boundary is still a cluster inside the returned window. Mark then return.
    _mark_tie_clusters(top)
    return top
