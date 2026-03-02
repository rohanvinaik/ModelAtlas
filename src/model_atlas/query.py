"""Navigational query engine for the semantic network.

Translates natural language queries into bank-position constraints and
anchor-similarity scores. Queries are navigational — exploring semantic
space, not running WHERE clauses.

Supports compound multi-bank queries, gradient proximity scoring,
and spreading activation from seed models.
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache

from . import db
from .config import (
    NAVIGATE_AVOID_DECAY,
    NAVIGATE_MISSING_BANK_PENALTY,
    WEIGHT_ANCHOR,
    WEIGHT_BANK,
    WEIGHT_FUZZY,
    WEIGHT_SPREAD,
)
from .query_types import (  # noqa: F401 — re-exported for backward compat
    BankConstraint,
    ComparisonResult,
    NavigationResult,
    ParsedQuery,
    SearchResult,
    StructuredQuery,
)
from .spreading import spread

# Query signal keywords mapped to bank constraints
_BANK_KEYWORDS: dict[str, list[tuple[str, int | None, int | None, int]]] = {
    # keyword -> [(bank, min_signed, max_signed, direction), ...]
    "small": [("EFFICIENCY", None, -1, -1)],
    "tiny": [("EFFICIENCY", None, -2, -1)],
    "large": [("EFFICIENCY", 1, None, 1)],
    "huge": [("EFFICIENCY", 2, None, 1)],
    "frontier": [("EFFICIENCY", 3, None, 1)],
    "base": [("LINEAGE", -1, 0, -1)],
    "fine-tune": [("LINEAGE", 1, None, 1)],
    "finetune": [("LINEAGE", 1, None, 1)],
    "derivative": [("LINEAGE", 2, None, 1)],
    "trending": [("QUALITY", 1, None, 1)],
    "popular": [("QUALITY", 1, None, 1)],
    "novel": [("ARCHITECTURE", 1, None, 1)],
    "specialized": [("DOMAIN", 1, None, 1)],
    "general": [("DOMAIN", -1, 0, -1)],
}

_ANCHOR_KEYWORDS: dict[str, str] = {
    "instruct": "instruction-following",
    "instruction": "instruction-following",
    "chat": "chat",
    "code": "code-generation",
    "coder": "code-generation",
    "tool": "tool-calling",
    "function": "function-calling",
    "reason": "reasoning",
    "math": "math",
    "vision": "image-understanding",
    "multimodal": "multimodal",
    "embed": "embedding",
    "gguf": "GGUF-available",
    "mlx": "MLX-compatible",
    "apple": "Apple-Silicon-native",
    "llama": "Llama-family",
    "mistral": "Mistral-family",
    "qwen": "Qwen-family",
    "phi": "Phi-family",
    "gemma": "Gemma-family",
    "deepseek": "DeepSeek-family",
    "medical": "medical-domain",
    "legal": "legal-domain",
    "finance": "finance-domain",
    "science": "science-domain",
}

# Pattern for "models like X" seed extraction
_LIKE_PATTERN = re.compile(
    r"(?:like|similar\s+to|related\s+to)\s+([a-zA-Z0-9/_.-]+)", re.IGNORECASE
)


@lru_cache(maxsize=1024)
def _parse_query(query: str) -> ParsedQuery:
    """Parse a natural language query into structured components."""
    tokens = re.findall(r"[a-zA-Z0-9_.-]+", query.lower())
    constraints: list[BankConstraint] = []
    anchor_targets: list[str] = []
    direction_vectors: dict[str, int] = {}

    for token in tokens:
        if token in _BANK_KEYWORDS:
            for bank, min_s, max_s, direction in _BANK_KEYWORDS[token]:
                constraints.append(
                    BankConstraint(
                        bank=bank,
                        direction=direction,
                        min_signed=min_s,
                        max_signed=max_s,
                    )
                )
                direction_vectors[bank] = direction
        if token in _ANCHOR_KEYWORDS:
            anchor_targets.append(_ANCHOR_KEYWORDS[token])

    # Seed model extraction ("models like meta-llama/Llama-3.1-8B")
    seed_ids: list[str] = []
    for match in _LIKE_PATTERN.finditer(query):
        seed_ids.append(match.group(1))

    return ParsedQuery(
        bank_constraints=constraints,
        anchor_targets=anchor_targets,
        seed_model_ids=seed_ids,
        direction_vectors=direction_vectors,
        raw_tokens=tokens,
    )


def _gradient_decay(distance: int | float) -> float:
    """Score that decays with distance: 1.0 / (1.0 + abs(distance))."""
    return 1.0 / (1.0 + abs(distance))


def _score_constraint(signed: int, c: BankConstraint) -> float:
    """Score a single constraint against a signed bank position."""
    lo = c.min_signed
    hi = c.max_signed

    if lo is not None and hi is not None:
        if lo <= signed <= hi:
            return 1.0
        return _gradient_decay(min(abs(signed - lo), abs(signed - hi)))

    if lo is not None:
        return 1.0 if signed >= lo else _gradient_decay(lo - signed)

    if hi is not None:
        return 1.0 if signed <= hi else _gradient_decay(signed - hi)

    if c.direction is not None:
        if signed == 0:
            return 0.5
        return 1.0 if (signed * c.direction > 0) else _gradient_decay(abs(signed))

    return 0.5


def _bank_proximity_score(
    model_positions: dict[str, dict],
    constraints: list[BankConstraint],
) -> float:
    """Score how well a model's bank positions match query constraints.

    Uses gradient proximity instead of binary match:
    - For range constraints: 1.0 / (1.0 + distance_from_range)
    - For directional queries: full score on desired side, decaying on wrong side
    """
    if not constraints:
        return 0.5

    scores: list[float] = []
    for c in constraints:
        pos = model_positions.get(c.bank)
        if not pos:
            scores.append(0.0)
            continue
        scores.append(_score_constraint(pos["sign"] * pos["depth"], c))

    return sum(scores) / len(scores) if scores else 0.5


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _confidence_weighted_jaccard(
    target_labels: set[str],
    model_anchors: list[dict],
) -> float:
    """Jaccard similarity weighted by anchor confidence.

    When model_anchors have 'confidence' values, weights each anchor
    by its confidence. Falls back to standard Jaccard for unweighted anchors.
    """
    if not target_labels and not model_anchors:
        return 0.0

    model_labels = {a["label"] for a in model_anchors}
    shared = target_labels & model_labels
    union = target_labels | model_labels

    if not union:
        return 0.0

    # Check if any anchors have confidence values
    conf_map = {a["label"]: a.get("confidence", 1.0) for a in model_anchors}
    # Target anchors get confidence 1.0 (they're what the user asked for)

    shared_weight = sum(conf_map.get(label, 1.0) for label in shared)
    union_weight = sum(conf_map.get(label, 1.0) for label in union)

    return shared_weight / union_weight if union_weight else 0.0


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    fuzzy_scores: dict[str, float] | None = None,
) -> list[SearchResult]:
    """Compound navigational search across the semantic network.

    Combines:
    1. Bank constraints (gradient scoring per bank, multiplied across banks)
    2. Anchor overlap (confidence-weighted Jaccard on target anchors)
    3. Spreading activation from seed models
    4. Fuzzy name-matching scores from Layer 2

    Score = WEIGHT_BANK * bank + WEIGHT_ANCHOR * anchor
          + WEIGHT_SPREAD * spread + WEIGHT_FUZZY * fuzzy
    """
    parsed = _parse_query(query)
    hint_set = set(parsed.anchor_targets)

    # Get all models
    all_models = conn.execute("SELECT model_id, author FROM models").fetchall()
    if not all_models:
        return []

    # Run spreading activation if we have seed models
    spread_scores: dict[str, float] = {}
    if parsed.seed_model_ids:
        # Scope spreading to the banks referenced in the query
        spread_banks = list(parsed.direction_vectors.keys()) or None
        spread_scores = spread(conn, parsed.seed_model_ids, banks=spread_banks)

    results: list[SearchResult] = []
    for row in all_models:
        mid = row["model_id"]
        model_data = db.get_model(conn, mid)
        if not model_data:
            continue

        positions = model_data.get("positions", {})
        model_anchor_list = model_data.get("anchors", [])
        model_anchor_labels = {a["label"] for a in model_anchor_list}

        # Bank proximity score (gradient)
        bank_score = _bank_proximity_score(positions, parsed.bank_constraints)

        # Anchor overlap score (confidence-weighted)
        if hint_set:
            anchor_score = _confidence_weighted_jaccard(hint_set, model_anchor_list)
        else:
            anchor_score = len(model_anchor_labels) / 50.0
            anchor_score = min(anchor_score, 1.0)

        # Spreading activation score
        s_score = spread_scores.get(mid, 0.0)

        # Fuzzy score (from external Layer 2)
        f_score = (fuzzy_scores or {}).get(mid, 0.0)

        # Combined score
        combined = (
            WEIGHT_BANK * bank_score
            + WEIGHT_ANCHOR * anchor_score
            + WEIGHT_SPREAD * s_score
            + WEIGHT_FUZZY * f_score
        )

        # Vibe summary from metadata
        vibe = model_data.get("metadata", {}).get("vibe_summary", {})
        vibe_text = vibe.get("value", "") if isinstance(vibe, dict) else ""

        results.append(
            SearchResult(
                model_id=mid,
                score=combined,
                bank_score=bank_score,
                anchor_score=anchor_score,
                spread_score=s_score,
                fuzzy_score=f_score,
                positions=positions,
                anchor_labels=sorted(model_anchor_labels),
                vibe_summary=vibe_text,
                author=row["author"] or "",
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


def similar_to(
    conn: sqlite3.Connection, model_id: str, limit: int = 10
) -> list[SearchResult]:
    """Find models similar to a given model via anchor Jaccard similarity."""
    target_anchors = db.get_anchor_set(conn, model_id)
    if not target_anchors:
        return []

    all_models = conn.execute(
        "SELECT model_id, author FROM models WHERE model_id != ?", (model_id,)
    ).fetchall()

    results: list[SearchResult] = []
    for row in all_models:
        mid = row["model_id"]
        other_anchors = db.get_anchor_set(conn, mid)
        score = _jaccard(target_anchors, other_anchors)
        if score > 0:
            results.append(
                SearchResult(
                    model_id=mid,
                    score=score,
                    anchor_score=score,
                    anchor_labels=sorted(target_anchors & other_anchors),
                    author=row["author"] or "",
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


def compare(conn: sqlite3.Connection, model_ids: list[str]) -> ComparisonResult:
    """Compare models via anchor set operations and bank position deltas."""
    anchor_sets: dict[str, set[str]] = {}
    position_maps: dict[str, dict] = {}

    for mid in model_ids:
        anchor_sets[mid] = db.get_anchor_set(conn, mid)
        model_data = db.get_model(conn, mid)
        position_maps[mid] = model_data.get("positions", {}) if model_data else {}

    # Shared anchors (intersection of all)
    if anchor_sets:
        shared = set.intersection(*anchor_sets.values()) if anchor_sets else set()
    else:
        shared = set()

    # Per-model unique anchors
    per_model_unique: dict[str, list[str]] = {}
    for mid, anchors in anchor_sets.items():
        per_model_unique[mid] = sorted(anchors - shared)

    # Jaccard similarity (pairwise average)
    jaccard_sum = 0.0
    pair_count = 0
    ids = list(anchor_sets.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            jaccard_sum += _jaccard(anchor_sets[ids[i]], anchor_sets[ids[j]])
            pair_count += 1
    avg_jaccard = jaccard_sum / pair_count if pair_count else 0.0

    # Bank position deltas
    bank_deltas: dict[str, dict] = {}
    for bank_name in db.BANKS:
        delta: dict[str, int] = {}
        for mid in model_ids:
            pos = position_maps.get(mid, {}).get(bank_name, {})
            delta[mid] = pos.get("sign", 0) * pos.get("depth", 0)
        bank_deltas[bank_name] = delta

    return ComparisonResult(
        models=model_ids,
        shared_anchors=sorted(shared),
        per_model_unique=per_model_unique,
        jaccard_similarity=round(avg_jaccard, 4),
        bank_deltas=bank_deltas,
    )


def lineage(conn: sqlite3.Connection, model_id: str) -> dict:
    """Traverse lineage relationships for a model.

    Results are ordered by signed LINEAGE position:
    predecessors (-N) first, base (0) in middle, derivatives (+N) at end.
    """
    model_data = db.get_model(conn, model_id)
    if not model_data:
        return {"model_id": model_id, "error": "not found in network"}

    links = model_data.get("links", {})
    lineage_pos = model_data.get("positions", {}).get("LINEAGE", {})

    derived_from = [
        link
        for link in links.get("outgoing", [])
        if link["relation"] in ("fine_tuned_from", "quantized_from", "variant_of")
    ]
    derivatives = [
        link
        for link in links.get("incoming", [])
        if link["relation"] in ("fine_tuned_from", "quantized_from", "variant_of")
    ]
    family = [
        link
        for link in links.get("outgoing", []) + links.get("incoming", [])
        if link["relation"] == "same_family"
    ]

    # Enrich with LINEAGE positions and sort
    def _get_lineage_signed(mid: str) -> int:
        pos = conn.execute(
            "SELECT path_sign, path_depth FROM model_positions WHERE model_id = ? AND bank = 'LINEAGE'",
            (mid,),
        ).fetchone()
        if pos:
            return pos[0] * pos[1]
        return 0

    def _enrich_link(link: dict) -> dict:
        # Determine which ID is the "other" model
        other_id = link.get("target_id") or link.get("source_id", "")
        return {**link, "lineage_signed": _get_lineage_signed(other_id)}

    derived_from = sorted(
        [_enrich_link(lnk) for lnk in derived_from],
        key=lambda x: x["lineage_signed"],
    )
    derivatives = sorted(
        [_enrich_link(lnk) for lnk in derivatives],
        key=lambda x: x["lineage_signed"],
    )
    family = sorted(
        [_enrich_link(lnk) for lnk in family],
        key=lambda x: x["lineage_signed"],
    )

    return {
        "model_id": model_id,
        "lineage_position": lineage_pos,
        "derived_from": derived_from,
        "derivatives": derivatives,
        "family": family,
    }


# ---------------------------------------------------------------------------
# Structured navigation — the calling LLM fills in StructuredQuery,
# ModelAtlas does deterministic math.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# navigate() scoring helpers — extracted to reduce cognitive complexity
# ---------------------------------------------------------------------------


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
