"""Tests to close remaining mutation gaps across the codebase.

Each test pins a specific constant or behavior that a mutant could change.
"""

from __future__ import annotations

from src.model_atlas._formatting import format_network_results
from src.model_atlas.db_queries import _in_clause
from src.model_atlas.extraction.vibes import (
    build_quality_gate_prompt,
    build_vibe_prompt,
    extract_vibe_summary,
)
from src.model_atlas.query_types import SearchResult
from src.model_atlas.sources.registry import get_source


# --- _in_clause: pin the separator and placeholder ---


def test_in_clause_single():
    assert _in_clause(1) == "?"


def test_in_clause_multiple():
    assert _in_clause(3) == "?,?,?"


def test_in_clause_separator_is_comma():
    result = _in_clause(2)
    assert "," in result
    assert result.count("?") == 2


# --- format_network_results: pin the anchor truncation at 15 ---


def test_format_network_results_truncates_anchors():
    r = SearchResult(
        model_id="test/model",
        score=1.0,
        bank_score=0.5,
        anchor_score=0.5,
        spread_score=0.0,
        fuzzy_score=0.0,
        positions={},
        anchor_labels=[f"anchor-{i}" for i in range(20)],
        vibe_summary="test",
        author="test",
    )
    result = format_network_results([r])
    assert len(result[0]["anchors"]) == 15


def test_format_network_results_rounding():
    r = SearchResult(
        model_id="test/model",
        score=0.12345,
        bank_score=0.6789,
        anchor_score=0.1111,
        spread_score=0.2222,
        fuzzy_score=0.3333,
        positions={},
        anchor_labels=[],
        vibe_summary="",
        author="a",
    )
    result = format_network_results([r])
    assert result[0]["score"] == 0.1235
    assert result[0]["score_breakdown"]["bank_proximity"] == 0.6789


# --- extract_vibe_summary: stub always returns empty string ---


def test_extract_vibe_summary_returns_empty():
    result = extract_vibe_summary("test/model", card_text="some text")
    assert result == ""


def test_extract_vibe_summary_ignores_all_params():
    result = extract_vibe_summary(
        "test/model",
        card_text="card",
        pipeline_tag="text-generation",
        tags=["llama"],
        author="meta",
        param_count="8B",
        family="Llama-family",
        capabilities=["chat"],
        training_method="rlhf",
    )
    assert result == ""


# --- build_vibe_prompt: pin template interpolation ---


def test_build_vibe_prompt_contains_model_id():
    result = build_vibe_prompt("meta-llama/Llama-3")
    assert "meta-llama/Llama-3" in result


def test_build_vibe_prompt_truncates_tags():
    tags = [f"tag-{i}" for i in range(20)]
    result = build_vibe_prompt("test/model", tags=tags)
    # Only first 15 tags should be in the output
    assert "tag-14" in result
    assert "tag-15" not in result


def test_build_vibe_prompt_defaults():
    result = build_vibe_prompt("test/model")
    assert "unknown" in result  # param_count default
    assert "none" in result  # tags default


# --- build_quality_gate_prompt: pin template ---


def test_build_quality_gate_prompt_contains_model_id():
    result = build_quality_gate_prompt("test/model", "A great model")
    assert "test/model" in result
    assert "A great model" in result


def test_build_quality_gate_prompt_truncates_tags():
    tags = [f"tag-{i}" for i in range(20)]
    result = build_quality_gate_prompt("test/model", "summary", tags=tags)
    assert "tag-14" in result
    assert "tag-15" not in result


# --- registry: get_source returns known sources ---


def test_get_source_huggingface():
    source = get_source("huggingface")
    assert source is not None
