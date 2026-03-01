"""Navigational query engine for the semantic network.

Translates natural language queries into bank-position constraints and
anchor-similarity scores. Queries are navigational — exploring semantic
space, not running WHERE clauses.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from . import db
from .config import WEIGHT_ANCHOR_OVERLAP, WEIGHT_BANK_PROXIMITY, WEIGHT_FUZZY


@dataclass
class SearchResult:
    """A model returned from a navigational search."""

    model_id: str
    score: float
    bank_score: float = 0.0
    anchor_score: float = 0.0
    fuzzy_score: float = 0.0
    positions: dict[str, dict] = field(default_factory=dict)
    anchor_labels: list[str] = field(default_factory=list)
    vibe_summary: str = ""
    author: str = ""


@dataclass
class ComparisonResult:
    """Result of comparing two or more models."""

    models: list[str]
    shared_anchors: list[str]
    per_model_unique: dict[str, list[str]]
    jaccard_similarity: float
    bank_deltas: dict[str, dict]


# Query signal keywords mapped to bank constraints and anchor hints
_BANK_KEYWORDS: dict[str, list[tuple[str, int | None, int | None]]] = {
    # keyword -> [(bank, min_signed, max_signed), ...]
    "small": [("EFFICIENCY", None, -1)],
    "tiny": [("EFFICIENCY", None, -2)],
    "large": [("EFFICIENCY", 1, None)],
    "huge": [("EFFICIENCY", 2, None)],
    "frontier": [("EFFICIENCY", 3, None)],
    "base": [("LINEAGE", -1, 0)],
    "fine-tune": [("LINEAGE", 1, None)],
    "finetune": [("LINEAGE", 1, None)],
    "derivative": [("LINEAGE", 2, None)],
    "trending": [("QUALITY", 1, None)],
    "popular": [("QUALITY", 1, None)],
    "novel": [("ARCHITECTURE", 1, None)],
    "specialized": [("DOMAIN", 1, None)],
    "general": [("DOMAIN", -1, 0)],
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


def _parse_query(
    query: str,
) -> tuple[list[tuple[str, int | None, int | None]], list[str]]:
    """Parse a natural language query into bank constraints and anchor hints."""
    tokens = re.findall(r"[a-zA-Z0-9_.-]+", query.lower())
    constraints: list[tuple[str, int | None, int | None]] = []
    anchor_hints: list[str] = []

    for token in tokens:
        if token in _BANK_KEYWORDS:
            constraints.extend(_BANK_KEYWORDS[token])
        if token in _ANCHOR_KEYWORDS:
            anchor_hints.append(_ANCHOR_KEYWORDS[token])

    return constraints, anchor_hints


def _bank_proximity_score(
    model_positions: dict[str, dict],
    constraints: list[tuple[str, int | None, int | None]],
) -> float:
    """Score how well a model's bank positions match query constraints."""
    if not constraints:
        return 0.5  # neutral when no constraints

    scores: list[float] = []
    for bank, min_s, max_s in constraints:
        pos = model_positions.get(bank)
        if not pos:
            scores.append(0.0)
            continue
        signed = pos["sign"] * pos["depth"]
        in_range = True
        if min_s is not None and signed < min_s:
            in_range = False
        if max_s is not None and signed > max_s:
            in_range = False
        scores.append(1.0 if in_range else 0.0)

    return sum(scores) / len(scores) if scores else 0.5


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    fuzzy_scores: dict[str, float] | None = None,
) -> list[SearchResult]:
    """Navigational search across the semantic network.

    Combines bank-position constraints with anchor-similarity scoring.
    Optionally incorporates fuzzy name-matching scores from Layer 2.
    """
    constraints, anchor_hints = _parse_query(query)
    hint_set = set(anchor_hints)

    # Get all models (for now; can optimize with bank-based pre-filtering)
    all_models = conn.execute("SELECT model_id, author FROM models").fetchall()
    if not all_models:
        return []

    results: list[SearchResult] = []
    for row in all_models:
        mid = row["model_id"]
        model_data = db.get_model(conn, mid)
        if not model_data:
            continue

        positions = model_data.get("positions", {})
        model_anchors = {a["label"] for a in model_data.get("anchors", [])}

        # Bank proximity score
        bank_score = _bank_proximity_score(positions, constraints)

        # Anchor overlap score
        if hint_set:
            anchor_score = _jaccard(hint_set, model_anchors)
        else:
            anchor_score = len(model_anchors) / 50.0  # normalize by typical max
            anchor_score = min(anchor_score, 1.0)

        # Fuzzy score (from external Layer 2)
        f_score = (fuzzy_scores or {}).get(mid, 0.0)

        # Combined score
        combined = (
            WEIGHT_BANK_PROXIMITY * bank_score
            + WEIGHT_ANCHOR_OVERLAP * anchor_score
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
                fuzzy_score=f_score,
                positions=positions,
                anchor_labels=sorted(model_anchors),
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
    """Traverse lineage relationships for a model."""
    model_data = db.get_model(conn, model_id)
    if not model_data:
        return {"model_id": model_id, "error": "not found in network"}

    links = model_data.get("links", {})
    lineage_pos = model_data.get("positions", {}).get("LINEAGE", {})

    return {
        "model_id": model_id,
        "lineage_position": lineage_pos,
        "derived_from": [
            link
            for link in links.get("outgoing", [])
            if link["relation"] in ("fine_tuned_from", "quantized_from", "variant_of")
        ],
        "derivatives": [
            link
            for link in links.get("incoming", [])
            if link["relation"] in ("fine_tuned_from", "quantized_from", "variant_of")
        ],
        "family": [
            link
            for link in links.get("outgoing", []) + links.get("incoming", [])
            if link["relation"] == "same_family"
        ],
    }
