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


NAVIGATE_OPPOSITION_PENALTY = 0.05
"""Monty Hall sharpening — a model whose position on a bank is DIRECTLY
opposed to the query's stated direction (e.g. model efficiency=+2 vs query
efficiency=-1) is confidently NOT what the user asked for. Old behavior
returned 0.33-0.5 (soft), letting opposed models place in top-N by other
signals. Sharpened: opposition → 0.05, so opposed models effectively drop
out of the top-N unless every other signal saturates. Not literal zero —
they remain visible in longer result sets, and MMR can still redistribute."""


def _bank_score_single(model_signed_pos: int, query_direction: int) -> float:
    """Score a single bank alignment.

    direction == 0: want near zero, penalize distance.
    direction == +1/-1: reward alignment, near-zero-out on opposition
    (Monty Hall — opposed models are confidently NOT the answer).
    """
    if query_direction == 0:
        return 1.0 / (1.0 + abs(model_signed_pos))
    alignment = model_signed_pos * query_direction
    if alignment > 0:
        return 1.0
    if alignment == 0:
        return 0.5
    return NAVIGATE_OPPOSITION_PENALTY


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


# ─────────────────────────────────────────────────────────────────────
# Structural additions — the "big blast" of principled scoring signals
# ─────────────────────────────────────────────────────────────────────

import math as _math

# Bank-family classification (dual where the bank carries both mechanical
# and semantic weight). Used by mechanical_fraction below.
_BANK_MECH_SEM: dict[str, tuple[float, float]] = {
    "EFFICIENCY": (1.0, 0.0),
    "COMPATIBILITY": (1.0, 0.0),
    "LINEAGE": (1.0, 0.0),
    "ARCHITECTURE": (1.0, 1.0),  # both — measured structure + intent
    "CAPABILITY": (0.0, 1.0),
    "DOMAIN": (0.0, 1.0),
    "QUALITY": (0.0, 1.0),
    "TRAINING": (0.0, 1.0),
}

NAVIGATE_STANDARDS_THRESHOLD = 0.40
"""An anchor present in > this fraction of the query's candidate sub-corpus
is a "standard" for that query context — the default state, whose absence
carries information (a specialist is CONSPICUOUSLY missing it)."""

NAVIGATE_SUBMODULAR_DECAY = 0.7
"""Sorted-descending signals combine with each successive one × decay.
The strongest signal counts fully; the second × 0.7; the third × 0.49.
Prevents flat multiplicative compounding when multiple signals point the
same way (submodular / diminishing returns)."""

NAVIGATE_MMR_LAMBDA = 0.7
"""MMR trade-off: higher = relevance-heavy, lower = diversity-heavy."""


def _mechanical_fraction(
    query: "StructuredQuery", idf: dict[str, float], conn: sqlite3.Connection
) -> float:
    """Query's mechanical vs semantic character in [0, 1]. 1.0 = pure
    mechanical (size/format), 0.0 = pure semantic (capability/domain).
    Drives mode='auto' weight selection. Anchor contributions are IDF-
    weighted so a rare semantic anchor pushes semantic-side harder than
    a common one would."""
    mech = sem = 0.0
    for bank in query.bank_directions().keys():
        m, s = _BANK_MECH_SEM.get(bank, (0.5, 0.5))
        mech += m
        sem += s
    for anchor in list(query.require_anchors) + list(query.prefer_anchors):
        row = conn.execute(
            "SELECT bank FROM anchors WHERE label = ?", (anchor,)
        ).fetchone()
        if row:
            m, s = _BANK_MECH_SEM.get(row["bank"], (0.5, 0.5))
            w = idf.get(anchor, 1.0)
            mech += m * w
            sem += s * w
    total = mech + sem
    return mech / total if total > 0 else 0.5


def _mode_weights(mode: str, mech_frac: float) -> dict[str, float]:
    """Weight multipliers for each soft signal, driven by query mode.

    * canonical → PageRank dominates (mechanical/basic query pattern)
    * niche → rare/absence/super dominate (specialist search)
    * balanced → fixed defaults
    * auto → linearly interpolate based on mechanical_fraction

    Return keys: K_PMI, K_RARE, K_ABS, K_SUPER, K_PR, MMR_LAMBDA.
    """
    if mode == "canonical":
        return {"K_PMI": 0.10, "K_RARE": 0.10, "K_ABS": 0.10,
                "K_SUPER": 0.10, "K_PR": 0.30, "MMR_LAMBDA": 0.9}
    if mode == "niche":
        return {"K_PMI": 0.40, "K_RARE": 0.50, "K_ABS": 0.40,
                "K_SUPER": 0.30, "K_PR": 0.05, "MMR_LAMBDA": 0.5}
    if mode == "balanced":
        return {"K_PMI": 0.25, "K_RARE": 0.25, "K_ABS": 0.25,
                "K_SUPER": 0.20, "K_PR": 0.20, "MMR_LAMBDA": 0.7}
    # auto — derive from mechanical fraction
    sem_frac = 1.0 - mech_frac
    return {
        "K_PMI": 0.20 + 0.20 * sem_frac,
        "K_RARE": 0.20 + 0.30 * sem_frac,
        "K_ABS": 0.15 + 0.25 * sem_frac,
        "K_SUPER": 0.15 + 0.15 * sem_frac,
        "K_PR": NAVIGATE_PAGERANK_MAX_BOOST + 0.20 * mech_frac,
        "MMR_LAMBDA": 0.85 - 0.35 * sem_frac,
    }


def _pmi_map(
    conn: sqlite3.Connection, candidate_ids: list[str], corpus_total: int
) -> dict[str, float]:
    """`{anchor: PMI(anchor, candidates)}` for anchors positively correlated
    with the query's candidate set relative to the corpus."""
    if not candidate_ids:
        return {}
    ph = ",".join("?" for _ in candidate_ids)
    cand_rows = conn.execute(
        f"""SELECT a.label, COUNT(DISTINCT ma.model_id) as n
              FROM anchors a JOIN model_anchors ma ON a.anchor_id = ma.anchor_id
             WHERE ma.model_id IN ({ph})
             GROUP BY a.anchor_id""",
        candidate_ids,
    ).fetchall()
    if not cand_rows:
        return {}
    corpus_rows = conn.execute(
        """SELECT a.label, COUNT(DISTINCT ma.model_id) as n
             FROM anchors a JOIN model_anchors ma ON a.anchor_id = ma.anchor_id
             GROUP BY a.anchor_id"""
    ).fetchall()
    corpus = {r["label"]: int(r["n"]) for r in corpus_rows}
    cand_total = len(candidate_ids)
    out: dict[str, float] = {}
    for row in cand_rows:
        n_g = corpus.get(row["label"], 0)
        if n_g == 0:
            continue
        pmi = _math.log((int(row["n"]) / cand_total) / (n_g / corpus_total))
        if pmi > 0:
            out[row["label"]] = pmi
    return out


def _standards_and_probs(
    conn: sqlite3.Connection,
    candidate_ids: list[str],
    exclude: set[str],
    threshold: float = NAVIGATE_STANDARDS_THRESHOLD,
) -> dict[str, float]:
    """Anchors present in > `threshold` of the candidate sub-corpus, excluding
    `exclude` (typically the require_anchors, whose absence is impossible).
    Returned as `{anchor: P(anchor | sub-corpus)}` — the P value drives the
    absence-bonus (-log(P) is the surprise mass of being missing)."""
    if not candidate_ids:
        return {}
    ph = ",".join("?" for _ in candidate_ids)
    rows = conn.execute(
        f"""SELECT a.label, COUNT(DISTINCT ma.model_id) as n
              FROM anchors a JOIN model_anchors ma ON a.anchor_id = ma.anchor_id
             WHERE ma.model_id IN ({ph})
             GROUP BY a.anchor_id""",
        candidate_ids,
    ).fetchall()
    cand_total = len(candidate_ids)
    return {
        r["label"]: int(r["n"]) / cand_total
        for r in rows
        if r["label"] not in exclude and int(r["n"]) / cand_total > threshold
    }


def _nav_rare_boost(
    model_anchors: set[str],
    prefer_set: set[str],
    pmi: dict[str, float],
) -> float:
    """IDF-weighted (via PMI proxy) match strength on prefer_anchors.

    Returns fraction in [0, 1]: matched-prefer PMI mass / total-prefer PMI
    mass. A model matching rare high-PMI prefer anchors scores high; matching
    common low-PMI ones scores low. Absent prefer set → 0 (neutral)."""
    if not prefer_set:
        return 0.0
    total_pmi = sum(pmi.get(a, 0.0) for a in prefer_set)
    if total_pmi <= 0:
        return 0.0
    matched_pmi = sum(pmi.get(a, 0.0) for a in (model_anchors & prefer_set))
    return matched_pmi / total_pmi


def _nav_absence_bonus(
    model_anchors: set[str], standards: dict[str, float]
) -> float:
    """Sum of -log(P(s | candidates)) for standards this model is CONSPICUOUSLY
    missing. High when the model diverges from the sub-corpus default state
    (specialist signal). Naturally accumulates for well-characterized
    specialists; sparse-anchor false positives are a known failure mode of
    this signal at the corpus-quality layer, not the scoring layer."""
    return sum(
        -_math.log(p) for s, p in standards.items() if s not in model_anchors
    )


def _submodular_combine(signals: list[float], decay: float = NAVIGATE_SUBMODULAR_DECAY) -> float:
    """Combine soft-signal deltas with diminishing returns.

    Returns `1 + Σ signal_i × decay^rank_i` where rank is descending by
    signal magnitude. The strongest signal counts fully; the second by
    `decay`; the third by `decay^2`. Prevents flat multiplicative
    compounding when multiple signals fire the same way (which double-
    counts what is really one piece of information)."""
    if not signals:
        return 1.0
    sorted_desc = sorted(signals, reverse=True)
    return 1.0 + sum(sig * (decay ** i) for i, sig in enumerate(sorted_desc))


def _mmr_rerank(
    results: list[NavigationResult],
    anchor_sets: dict[str, set[str]],
    lam: float = NAVIGATE_MMR_LAMBDA,
    top_k: int | None = None,
) -> list[NavigationResult]:
    """Maximum Marginal Relevance re-rank. Greedy pick that maximizes
    `lam × relevance - (1-lam) × max_similarity_to_already_selected`.
    Similarity = Jaccard over model anchor sets. Preserves the top-K
    return-count; only reorders within it to reduce near-duplicate stacking
    (e.g. finbert-tone next to finbert-tone-chinese)."""
    if len(results) <= 1:
        return results
    k = min(top_k or len(results), len(results))
    pool = list(results)
    selected: list[NavigationResult] = []
    while pool and len(selected) < k:
        best = None
        best_score = -float("inf")
        for cand in pool:
            rel = cand.score
            if not selected:
                mmr_score = rel
            else:
                cand_anchors = anchor_sets.get(cand.model_id, set())
                max_sim = 0.0
                for s in selected:
                    s_anchors = anchor_sets.get(s.model_id, set())
                    if cand_anchors and s_anchors:
                        sim = len(cand_anchors & s_anchors) / len(cand_anchors | s_anchors)
                        if sim > max_sim:
                            max_sim = sim
                mmr_score = lam * rel - (1 - lam) * max_sim * rel
            if mmr_score > best_score:
                best_score = mmr_score
                best = cand
        if best is None:
            break
        selected.append(best)
        pool.remove(best)
    return selected


def _apply_bank_weights(
    directions: dict[str, int],
    bank_weights: dict[str, float] | None,
) -> dict[str, float]:
    """Return per-bank exponents to apply to `_bank_score_single` results.

    Missing bank in `bank_weights` → 1.0 (default). Zero weight → 0 (bank
    neutralized, its contribution becomes `x^0 = 1`, i.e. no effect on
    the score). Renormalized across the ACTIVE (non-zero) banks so that
    the total attention-mass equals N_active — a bank explicitly weighted
    2× takes proportional mass from the others, preserving the overall
    scale of bank_alignment. Missing bank_weights arg (None) → all 1.0
    (current behavior, no change)."""
    if not bank_weights or not directions:
        return {bank: 1.0 for bank in directions}
    raw = {bank: max(0.0, float(bank_weights.get(bank, 1.0))) for bank in directions}
    active_mass = sum(w for w in raw.values() if w > 0)
    active_count = sum(1 for w in raw.values() if w > 0)
    if active_mass <= 0 or active_count == 0:
        return {bank: 1.0 for bank in directions}
    scale = active_count / active_mass
    return {bank: (w * scale if w > 0 else 0.0) for bank, w in raw.items()}


def _nav_bank_alignment_weighted(
    positions: dict[str, tuple[int, int]],
    directions: dict[str, int],
    bank_weights: dict[str, float],
) -> float:
    """Bank alignment product with per-bank exponent weights. Bank with
    weight 0 becomes `x^0 = 1` (neutralized, no contribution)."""
    if not directions:
        return 1.0
    result = 1.0
    for bank_name, direction in directions.items():
        pos = positions.get(bank_name)
        base = NAVIGATE_MISSING_BANK_PENALTY if pos is None else _bank_score_single(pos[0] * pos[1], direction)
        w = bank_weights.get(bank_name, 1.0)
        if w > 0:
            result *= base ** w
        # w == 0 → factor becomes 1.0 (bank neutralized)
    return result


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
    _pr_max = max(pagerank_by_id.values(), default=0.0)

    # Per-query preparation for the new signals (PMI, standards, mode, bank
    # weights). All cheap: one anchor-count query for PMI/standards, pure
    # Python for the rest. Basic queries with no rare anchors / no clear
    # standards / mode='auto' produce empty PMI + empty standards + balanced
    # weights → the new signals reduce to their neutral values (0), and
    # `_submodular_combine([0,0,0,0,0]) == 1.0` — behavior identical to
    # pre-refactor. Structural correctness without changing basic outcomes.
    pmi = _pmi_map(conn, candidate_ids, corpus_total=_corpus_total(conn))
    standards = _standards_and_probs(conn, candidate_ids, exclude=require_set)
    mech_frac = _mechanical_fraction(query, idf, conn)
    mode_w = _mode_weights(query.mode or "auto", mech_frac)
    bank_exp = _apply_bank_weights(directions, query.bank_weights)
    query_anchors_pool = require_set | prefer_set
    max_pmi_match = sum(pmi.get(a, 0.0) for a in query_anchors_pool) or 1.0
    max_absence = sum(-_math.log(p) for p in standards.values()) or 1.0

    # Step 3: Score each candidate
    results: list[NavigationResult] = []
    for mid in candidate_ids:
        positions = all_positions.get(mid, {})
        model_anchor_set = all_anchors.get(mid, set())

        # ── Bank alignment with per-bank weight exponents (bank_weights) ──
        bank_alignment = _nav_bank_alignment_weighted(positions, directions, bank_exp)
        anchor_relevance = _nav_anchor_relevance(
            model_anchor_set, prefer_set, avoid_set, idf, prefer_idf_total, has_anchor_constraints,
        )
        seed_similarity = _nav_seed_similarity(model_anchor_set, seed_anchors, idf)
        coherence = coherence_by_id.get(mid, 1.0)
        context_bias = _nav_context_bias(model_anchor_set, context_set, idf)
        epa_alignment = _nav_epa_alignment(
            epa_by_id.get(mid), (query.vibe_e, query.vibe_p, query.vibe_a)
        )

        # ── Soft signals combined submodularly (diminishing returns) ──
        pr_frac = (pagerank_by_id.get(mid, 0.0) / _pr_max) if _pr_max > 0 else 0.0
        pmi_match = (
            sum(pmi.get(a, 0.0) for a in (model_anchor_set & query_anchors_pool)) / max_pmi_match
        )
        rare_boost = _nav_rare_boost(model_anchor_set, prefer_set, pmi)
        absence = _nav_absence_bonus(model_anchor_set, standards) / max_absence
        super_ = pr_frac * rare_boost  # joint = superadditive when both fire

        soft_signals = [
            mode_w["K_PR"] * pr_frac,
            mode_w["K_PMI"] * pmi_match,
            mode_w["K_RARE"] * rare_boost,
            mode_w["K_ABS"] * absence,
            mode_w["K_SUPER"] * super_,
        ]
        soft_combined = _submodular_combine(soft_signals)
        pagerank_boost_alone = 1.0 + mode_w["K_PR"] * pr_frac  # what PR alone contributed, for surfacing

        final_score = (
            bank_alignment
            * anchor_relevance
            * seed_similarity
            * coherence
            * context_bias
            * epa_alignment
            * soft_combined
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
                pagerank_boost=pagerank_boost_alone,
                soft_combined=soft_combined,
                positions=pos_out,
                anchor_labels=sorted(model_anchor_set),
                author=authors.get(mid, ""),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    top = results[: query.limit]
    # ── MMR diversify within the returned window ──
    anchor_sets_by_id = {r.model_id: set(r.anchor_labels) for r in top}
    top = _mmr_rerank(top, anchor_sets_by_id, lam=mode_w["MMR_LAMBDA"], top_k=len(top))
    # Cluster AFTER MMR so tie_cluster reflects the actual returned order.
    _mark_tie_clusters(top)
    return top


def _corpus_total(conn: sqlite3.Connection) -> int:
    """Total model count for PMI base rate. Cheap indexed COUNT."""
    return int(conn.execute("SELECT COUNT(*) FROM models").fetchone()[0])
