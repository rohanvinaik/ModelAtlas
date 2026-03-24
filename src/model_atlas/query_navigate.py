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


def navigate(
    conn: sqlite3.Connection,
    query: StructuredQuery,
) -> list[NavigationResult]:
    """Structured navigational search — the primary recommendation engine.

    Three signals, multiplicative:
        final_score = bank_alignment * anchor_relevance * seed_similarity

    Uses batch SQL instead of N+1 per-model lookups.
    """
    idf = _get_idf(conn)
    directions = query.bank_directions()
    require_set = set(query.require_anchors)
    prefer_set = set(query.prefer_anchors)
    avoid_set = set(query.avoid_anchors)
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
        final_score = bank_alignment * anchor_relevance * seed_similarity

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
                positions=pos_out,
                anchor_labels=sorted(model_anchor_set),
                author=authors.get(mid, ""),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[: query.limit]
