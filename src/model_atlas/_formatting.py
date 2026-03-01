"""Response formatting helpers for the MCP server.

Converts internal data structures (SearchResult, StructuredResult, fuzzy scores)
into JSON-serializable dicts for MCP tool responses.
"""

from __future__ import annotations

import json

from .search.structured import StructuredResult


def structured_to_dict(r: StructuredResult) -> dict:
    """Convert a StructuredResult to a plain dict."""
    return {
        "model_id": r.model_id,
        "author": r.author,
        "likes": r.likes,
        "downloads": r.downloads,
        "pipeline_tag": r.pipeline_tag,
        "tags": r.tags,
        "library_name": r.library_name,
        "license": r.license,
        "card_text": r.card_text,
        "rank": r.rank,
    }


def candidates_to_dicts(candidates: list[StructuredResult]) -> list[dict]:
    """Convert structured results to dicts suitable for extract_batch."""
    return [
        {
            "model_id": c.model_id,
            "author": c.author,
            "pipeline_tag": c.pipeline_tag,
            "tags": c.tags,
            "library_name": c.library_name,
            "likes": c.likes,
            "downloads": c.downloads,
            "created_at": c.raw.get("created_at"),
            "license": c.license,
            "card_text": c.card_text,
        }
        for c in candidates
    ]


def format_network_results(network_results: list) -> list[dict]:
    """Format SearchResult objects for JSON output."""
    return [
        {
            "model_id": r.model_id,
            "author": r.author,
            "score": round(r.score, 4),
            "score_breakdown": {
                "bank_proximity": round(r.bank_score, 4),
                "anchor_overlap": round(r.anchor_score, 4),
                "fuzzy": round(r.fuzzy_score, 4),
            },
            "positions": r.positions,
            "anchors": r.anchor_labels[:15],
            "vibe": r.vibe_summary,
        }
        for r in network_results
    ]


def format_fuzzy_results(
    candidates: list[dict],
    fuzzy_scores: dict[str, float],
    limit: int,
) -> list[dict]:
    """Format results when network is empty (fuzzy-only fallback)."""
    scored = []
    for c in candidates:
        mid = c["model_id"]
        scored.append((fuzzy_scores.get(mid, 0.0), c))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "model_id": c["model_id"],
            "author": c["author"],
            "score": round(score, 4),
            "likes": c["likes"],
            "downloads": c["downloads"],
            "pipeline_tag": c["pipeline_tag"],
            "tags": c["tags"][:10] if c["tags"] else [],
            "library": c["library_name"],
        }
        for score, c in scored[:limit]
    ]


def fetch_from_hf_api(model_id: str) -> str:
    """Fetch model details directly from HuggingFace API."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        info = api.model_info(model_id)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch model info: {e}"})

    return json.dumps(
        {
            "source": "huggingface_api",
            "model_id": info.id,
            "author": info.author,
            "likes": info.likes,
            "downloads": info.downloads,
            "pipeline_tag": info.pipeline_tag,
            "tags": list(info.tags or []),
            "library_name": info.library_name,
            "license": getattr(info, "license", None),
            "hint": "This model is not in the semantic network. "
            "Run hf_build_index to add it for richer navigational queries.",
        },
        indent=2,
        default=str,
    )
