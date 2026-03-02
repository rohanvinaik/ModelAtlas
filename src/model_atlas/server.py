"""ModelAtlas MCP Server.

Navigational search across a semantic network of ML models.
Run with: uv run model-atlas
"""

from __future__ import annotations

import json
import logging
import sqlite3

from mcp.server.fastmcp import FastMCP

from . import db
from ._formatting import (
    candidates_to_dicts,
    fetch_from_hf_api,
    format_fuzzy_results,
    format_network_results,
    structured_to_dict,
)
from .config import DEFAULT_CANDIDATE_LIMIT, DEFAULT_INDEX_SIZE, DEFAULT_RESULT_LIMIT
from .extraction.pipeline import extract_batch
from .query import StructuredQuery, compare, navigate, search
from .search import fuzzy, structured

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "model-atlas",
    instructions=(
        "Fuzzy and semantic search across HuggingFace Hub models. "
        "WORKFLOW: Always start with hf_search_models \u2014 it works immediately with structured+fuzzy layers. "
        "Check the 'semantic_index' field in results: if 'not_built' AND the query is subjective/vibe-based "
        "(not a keyword lookup), call hf_build_index for that category (one-time, 1-3 min), then re-search. "
        "Do NOT build indexes for keyword/factual queries \u2014 L1+L2 handle those. "
        "Do NOT build indexes speculatively for categories that haven't been queried. "
        "Indexes persist to disk and are reused across sessions."
    ),
)


def _ensure_db() -> None:
    """Initialize the database if it doesn't exist."""
    conn = db.get_connection()
    try:
        db.init_db(conn)
    finally:
        conn.close()


# Initialize on import
_ensure_db()


@mcp.tool()
def hf_search_models(
    query: str,
    task: str | None = None,
    author: str | None = None,
    library: str | None = None,
    min_likes: int = 0,
    min_downloads: int = 0,
    limit: int = DEFAULT_RESULT_LIMIT,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> str:
    """Search HuggingFace models using navigational semantic network + fuzzy matching.

    Combines three layers:
    - Network: Bank-position constraints + anchor similarity (if models indexed)
    - Fuzzy: String matching on model IDs, tags, descriptions
    - Structured: HF API filters for task, author, library, popularity

    Args:
        query: Natural language search query (e.g. "small code model with tool-calling")
        task: HuggingFace task filter (e.g. "text-generation", "text-classification")
        author: Filter by model author/org (e.g. "meta-llama", "mistralai")
        library: Filter by library (e.g. "transformers", "gguf", "diffusers")
        min_likes: Minimum number of likes
        min_downloads: Minimum number of downloads
        limit: Number of results to return (default 20)
        candidate_limit: How many candidates to fetch from HF API (default 500)
    """
    conn = db.get_connection()
    try:
        stats = db.network_stats(conn)
        has_network = stats["total_models"] > 0

        # Layer 1: Structured HF API search (always runs — source of live data)
        candidates = structured.search(
            task=task,
            search_query=query if not task else None,
            author=author,
            library=library,
            limit=candidate_limit,
            min_likes=min_likes,
            min_downloads=min_downloads,
        )
        candidate_dicts = [structured_to_dict(c) for c in candidates]

        # Layer 2: Fuzzy scoring
        fuzzy_results = fuzzy.score_models(query, candidate_dicts)
        fuzzy_scores = {f.model_id: f.score for f in fuzzy_results}

        if has_network:
            # Network search with fuzzy scores incorporated
            network_results = search(
                conn, query, limit=limit, fuzzy_scores=fuzzy_scores
            )
            results = format_network_results(network_results)
        else:
            # Fallback: rank by fuzzy scores alone
            results = format_fuzzy_results(candidate_dicts, fuzzy_scores, limit)

        output = {
            "query": query,
            "filters": {"task": task, "author": author, "library": library},
            "network_status": "active" if has_network else "empty",
            "network_models": stats["total_models"],
            "total_candidates": len(candidate_dicts),
            "result_count": len(results),
            "results": results,
        }
        if not has_network:
            output["hint"] = (
                "Semantic network is empty. Run hf_build_index to populate it "
                "for richer navigational search results."
            )
        return json.dumps(output, indent=2)
    finally:
        conn.close()


@mcp.tool()
def navigate_models(
    architecture: int | None = None,
    capability: int | None = None,
    efficiency: int | None = None,
    compatibility: int | None = None,
    lineage: int | None = None,
    domain: int | None = None,
    quality: int | None = None,
    require_anchors: list[str] | None = None,
    prefer_anchors: list[str] | None = None,
    avoid_anchors: list[str] | None = None,
    similar_to: str | None = None,
    limit: int = 20,
) -> str:
    """Navigate the semantic model network with structured scoring.

    The primary recommendation engine. Decompose the user's natural language
    query into these structured parameters — ModelAtlas does deterministic math.

    BANK DIRECTIONS (-1, 0, or +1 for each; omit to express "don't care"):

      architecture:  -1=simpler/older (encoder-only, RNN)
                      0=standard transformer decoder (most models)
                     +1=novel/specialized (Mamba, RWKV, SSM, MoE)

      capability:    -1=narrow/single-task (classifier, embedding-only)
                      0=general language model
                     +1=rich/multi-capability (code, reasoning, tool-calling)

      efficiency:    -1=smaller/lighter (3B, 1B, sub-1B, edge)
                      0=~7B mainstream sweet spot
                     +1=larger/frontier (13B, 30B, 70B+)

      compatibility:  0=standard PyTorch/transformers
                     +1=specific format/hardware (GGUF, MLX, Apple Silicon)

      lineage:       -1=predecessor/earlier in family
                      0=base/foundational model
                     +1=derivative (fine-tune, quantized, community merge)

      domain:         0=general knowledge
                     +1=domain-specialized (code, medical, legal, finance)

      quality:       -1=legacy/declining/abandoned
                      0=established mainstream
                     +1=trending/rising momentum

    ANCHOR TARGETING:
      require_anchors: Model MUST have ALL of these (hard filter).
        Examples: "code-generation", "instruction-following", "Llama-family",
                  "GGUF-available", "consumer-GPU-viable", "tool-calling"
      prefer_anchors: Boost models with these (soft preference, IDF-weighted).
      avoid_anchors: Penalize models with these (each halves the score).

    SEED SIMILARITY:
      similar_to: A model_id to use as similarity seed (IDF-weighted Jaccard
                  on anchor sets). Example: "meta-llama/Llama-3.1-8B-Instruct"

    Scoring: bank_alignment * anchor_relevance * seed_similarity
    Each ∈ [0,1]. Product means any zero kills the score.

    Args:
        architecture: Bank direction for ARCHITECTURE (-1, 0, or +1)
        capability: Bank direction for CAPABILITY (-1, 0, or +1)
        efficiency: Bank direction for EFFICIENCY (-1, 0, or +1)
        compatibility: Bank direction for COMPATIBILITY (-1, 0, or +1)
        lineage: Bank direction for LINEAGE (-1, 0, or +1)
        domain: Bank direction for DOMAIN (-1, 0, or +1)
        quality: Bank direction for QUALITY (-1, 0, or +1)
        require_anchors: Anchors the model MUST have (hard filter)
        prefer_anchors: Anchors that boost score (soft, IDF-weighted)
        avoid_anchors: Anchors that penalize score
        similar_to: Model ID for anchor-similarity seed
        limit: Max results to return (default 20)
    """
    conn = db.get_connection()
    try:
        stats = db.network_stats(conn)
        if stats["total_models"] == 0:
            return json.dumps(
                {
                    "error": "Semantic network is empty. Run hf_build_index first.",
                    "network_models": 0,
                }
            )

        sq = StructuredQuery(
            architecture=architecture,
            capability=capability,
            efficiency=efficiency,
            compatibility=compatibility,
            lineage=lineage,
            domain=domain,
            quality=quality,
            require_anchors=require_anchors or [],
            prefer_anchors=prefer_anchors or [],
            avoid_anchors=avoid_anchors or [],
            similar_to=similar_to,
            limit=limit,
        )
        results = navigate(conn, sq)

        return json.dumps(
            {
                "query": {
                    "banks": sq.bank_directions(),
                    "require_anchors": sq.require_anchors,
                    "prefer_anchors": sq.prefer_anchors,
                    "avoid_anchors": sq.avoid_anchors,
                    "similar_to": sq.similar_to,
                },
                "network_models": stats["total_models"],
                "result_count": len(results),
                "results": [
                    {
                        "model_id": r.model_id,
                        "author": r.author,
                        "score": round(r.score, 4),
                        "score_breakdown": {
                            "bank_alignment": round(r.bank_alignment, 4),
                            "anchor_relevance": round(r.anchor_relevance, 4),
                            "seed_similarity": round(r.seed_similarity, 4),
                        },
                        "positions": r.positions,
                        "anchors": r.anchor_labels[:15],
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    finally:
        conn.close()


@mcp.tool()
def hf_get_model_detail(model_id: str) -> str:
    """Get detailed information about a specific model.

    Returns full semantic network profile (all 7 bank positions, anchor set,
    lineage links, overflow metadata) if indexed. Falls back to HF API.

    Args:
        model_id: Full model ID (e.g. "meta-llama/Llama-3.1-8B-Instruct")
    """
    conn = db.get_connection()
    try:
        model_data = db.get_model(conn, model_id)
        if model_data:
            return json.dumps(
                {
                    "source": "network",
                    "model_id": model_data["model_id"],
                    "author": model_data["author"],
                    "positions": model_data["positions"],
                    "anchors": model_data["anchors"],
                    "links": model_data["links"],
                    "metadata": model_data["metadata"],
                },
                indent=2,
            )
    finally:
        conn.close()

    # Fallback: fetch from HF API
    return fetch_from_hf_api(model_id)


@mcp.tool()
def hf_compare_models(model_ids: list[str]) -> str:
    """Compare multiple models via anchor set operations and bank positions.

    Shows shared anchors (intersection), distinguishing features (unique per model),
    Jaccard similarity, and per-bank position deltas.

    Args:
        model_ids: List of model IDs to compare
    """
    conn = db.get_connection()
    try:
        result = compare(conn, model_ids)
        return json.dumps(
            {
                "models": result.models,
                "shared_anchors": result.shared_anchors,
                "per_model_unique_anchors": result.per_model_unique,
                "jaccard_similarity": result.jaccard_similarity,
                "bank_deltas": result.bank_deltas,
            },
            indent=2,
        )
    finally:
        conn.close()


@mcp.tool()
def hf_build_index(
    category: str,
    task: str | None = None,
    source: str = "huggingface",
    limit: int = DEFAULT_INDEX_SIZE,
    min_likes: int = 5,
    force: bool = False,
) -> str:
    """Fetch models and add them to the semantic network.

    Runs the full extraction pipeline: fetches models, extracts bank positions,
    anchor links, and metadata, then stores everything in the network database.
    This is additive — multiple calls enrich the same network.

    Args:
        category: Category label for this batch (e.g. "text-generation", "code")
        task: HuggingFace task filter to scope models. If None, uses category.
        source: Source to index from — 'huggingface' (default), 'ollama', or 'all'
        limit: Max models to fetch (default 2000)
        min_likes: Minimum likes threshold (default 5, only for huggingface)
        force: Currently unused; network is always additive
    """
    conn = db.get_connection()
    try:
        db.init_db(conn)

        if source == "ollama":
            return _build_index_ollama(conn, category, limit)
        elif source == "all":
            # Index from all available sources
            hf_result = _build_index_huggingface(conn, category, task, limit, min_likes)
            ollama_result = _build_index_ollama(conn, category, limit)
            stats = db.network_stats(conn)
            return json.dumps(
                {
                    "status": "indexed",
                    "category": category,
                    "sources": {"huggingface": hf_result, "ollama": ollama_result},
                    "network_total": stats["total_models"],
                    "network_anchors": stats["total_anchors"],
                }
            )
        else:
            return _build_index_huggingface(conn, category, task, limit, min_likes)
    finally:
        conn.close()


def _build_index_huggingface(
    conn: sqlite3.Connection,
    category: str,
    task: str | None,
    limit: int,
    min_likes: int,
) -> str:
    """Index models from HuggingFace."""
    search_task = task or category
    logger.info(
        "Fetching up to %d models for '%s' (task=%s)", limit, category, search_task
    )

    candidates = structured.search(
        task=search_task,
        sort="likes",
        limit=limit,
        min_likes=min_likes,
    )
    if not candidates:
        return json.dumps({"error": f"No models found for task '{search_task}'"})

    model_dicts = candidates_to_dicts(candidates)
    count = extract_batch(conn, model_dicts)
    conn.commit()
    stats = db.network_stats(conn)

    # Identify models needing vibes
    vibes_needed = _find_models_without_vibes(
        conn, [m["model_id"] for m in model_dicts]
    )

    result = {
        "status": "indexed",
        "category": category,
        "source": "huggingface",
        "models_fetched": len(candidates),
        "models_indexed": count,
        "network_total": stats["total_models"],
        "network_anchors": stats["total_anchors"],
    }
    if vibes_needed:
        result["vibes_needed"] = vibes_needed[:20]
        result["vibes_hint"] = (
            "These models lack vibe summaries. Use set_model_vibe to add them."
        )
    return json.dumps(result)


def _build_index_ollama(
    conn: sqlite3.Connection,
    category: str,
    limit: int,
) -> str:
    """Index models from local Ollama instance."""
    from .sources import get_source

    try:
        adapter = get_source("ollama")
    except KeyError:
        return json.dumps({"error": "Ollama source not available"})

    try:
        results = adapter.search("", limit=limit)
    except Exception as e:
        return json.dumps({"error": f"Ollama not reachable: {e}"})

    if not results:
        return json.dumps({"status": "no_models", "source": "ollama"})

    count = 0
    for r in results:
        try:
            inp = adapter.get_detail(r.model_id)
            from .extraction.pipeline import extract_and_store

            extract_and_store(conn, inp)
            count += 1
        except Exception:
            logger.warning(
                "Failed to index Ollama model: %s", r.model_id, exc_info=True
            )

    conn.commit()
    stats = db.network_stats(conn)

    return json.dumps(
        {
            "status": "indexed",
            "category": category,
            "source": "ollama",
            "models_fetched": len(results),
            "models_indexed": count,
            "network_total": stats["total_models"],
            "network_anchors": stats["total_anchors"],
        }
    )


def _find_models_without_vibes(
    conn: sqlite3.Connection, model_ids: list[str]
) -> list[str]:
    """Find model IDs that lack a vibe_summary in metadata."""
    if not model_ids:
        return []
    placeholders = ",".join("?" for _ in model_ids)
    rows = conn.execute(
        f"""SELECT model_id FROM model_metadata
           WHERE key = 'vibe_summary' AND model_id IN ({placeholders})""",
        model_ids,
    ).fetchall()
    have_vibes = {r[0] for r in rows}
    return [mid for mid in model_ids if mid not in have_vibes]


@mcp.tool()
def hf_index_status() -> str:
    """Show the current state of the semantic network.

    Returns total models, anchor dictionary size, per-bank breakdowns,
    and source coverage.
    """
    conn = db.get_connection()
    try:
        db.init_db(conn)
        stats = db.network_stats(conn)
        return json.dumps(stats, indent=2)
    finally:
        conn.close()


@mcp.tool()
def search_models(
    query: str,
    source: str = "huggingface",
    limit: int = 20,
    task: str | None = None,
    author: str | None = None,
) -> str:
    """Search models across sources (huggingface, ollama, or all).

    Uses source adapters to search model registries. When source='all',
    searches all registered sources and merges results.

    Args:
        query: Search query string
        source: Source to search — 'huggingface', 'ollama', or 'all'
        limit: Maximum results to return (default 20)
        task: Task filter (only applies to sources that support it)
        author: Author filter (only applies to sources that support it)
    """
    from .sources import get_source
    from .sources import list_sources as _list_sources

    filters = {}
    if task:
        filters["task"] = task
    if author:
        filters["author"] = author

    if source == "all":
        all_results = []
        errors = {}
        for name, adapter in _list_sources().items():
            try:
                results = adapter.search(query, limit=limit, filters=filters)
                all_results.extend(
                    {
                        "model_id": r.model_id,
                        "source": r.source,
                        "author": r.author,
                        "display_name": r.display_name,
                        "downloads": r.downloads,
                        "likes": r.likes,
                        "tags": r.tags[:10],
                    }
                    for r in results
                )
            except Exception as e:
                errors[name] = str(e)

        output = {
            "query": query,
            "source": "all",
            "result_count": len(all_results),
            "results": all_results[:limit],
        }
        if errors:
            output["errors"] = errors
        return json.dumps(output, indent=2)

    try:
        adapter = get_source(source)
    except KeyError:
        available = list(_list_sources().keys())
        return json.dumps(
            {"error": f"Unknown source '{source}'. Available: {available}"}
        )

    results = adapter.search(query, limit=limit, filters=filters)
    output = {
        "query": query,
        "source": source,
        "result_count": len(results),
        "results": [
            {
                "model_id": r.model_id,
                "source": r.source,
                "author": r.author,
                "display_name": r.display_name,
                "downloads": r.downloads,
                "likes": r.likes,
                "tags": r.tags[:10],
            }
            for r in results
        ],
    }
    return json.dumps(output, indent=2)


@mcp.tool()
def set_model_vibe(
    model_id: str,
    vibe_summary: str,
    extra_anchors: list[str] | None = None,
) -> str:
    """Set the vibe summary and optional extra anchors for a model.

    Called by the LLM after reading a model card or understanding a model's
    characteristics. The LLM IS the NLP extraction tier — delegate vibes
    extraction to the calling model rather than building NLP in Python.

    Args:
        model_id: Full model ID (e.g. "meta-llama/Llama-3.1-8B-Instruct")
        vibe_summary: One sentence capturing the model's distinctive feel
        extra_anchors: Optional anchor labels to add (e.g. ["tool-calling", "reasoning"])
    """
    conn = db.get_connection()
    try:
        # Verify model exists
        model_data = db.get_model(conn, model_id)
        if not model_data:
            return json.dumps({"error": f"Model '{model_id}' not found in network"})

        # Store vibe summary
        db.set_metadata(conn, model_id, "vibe_summary", vibe_summary, "str")

        # Add extra anchors with vibes provenance
        anchors_added = []
        if extra_anchors:
            for label in extra_anchors:
                # Infer bank from existing anchor vocabulary, default to CAPABILITY
                existing = conn.execute(
                    "SELECT bank FROM anchors WHERE label = ?", (label,)
                ).fetchone()
                bank = existing["bank"] if existing else "CAPABILITY"
                anchor_id = db.get_or_create_anchor(conn, label, bank, source="vibes")
                db.link_anchor(conn, model_id, anchor_id, confidence=0.5)
                anchors_added.append(label)

        conn.commit()
        return json.dumps(
            {
                "status": "updated",
                "model_id": model_id,
                "vibe_summary": vibe_summary,
                "anchors_added": anchors_added,
            }
        )
    finally:
        conn.close()


@mcp.tool()
def list_model_sources() -> str:
    """List available model sources and their status.

    Returns all registered source adapters with availability info.
    """
    from .sources import list_sources as _list_sources

    sources = {}
    for name, adapter in _list_sources().items():
        status = "available"
        # Quick availability check for non-HF sources
        if name == "ollama":
            try:
                adapter.search("", limit=1)
            except Exception:
                status = "unavailable (not running)"
        sources[name] = {"name": name, "status": status}

    return json.dumps({"sources": sources}, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
