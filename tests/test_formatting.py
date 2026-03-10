"""Tests for response formatting helpers."""

from __future__ import annotations

from model_atlas._formatting import (
    candidates_to_dicts,
    format_fuzzy_results,
    format_network_results,
    structured_to_dict,
)
from model_atlas.query_types import SearchResult
from model_atlas.search.structured import StructuredResult


def _make_structured(
    model_id: str = "test/Model-7B",
    author: str = "test",
    likes: int = 100,
    downloads: int = 5000,
    pipeline_tag: str = "text-generation",
    tags: list[str] | None = None,
    library_name: str = "transformers",
    license_str: str = "apache-2.0",
    card_text: str = "A test model.",
    rank: int = 0,
    raw: dict | None = None,
) -> StructuredResult:
    return StructuredResult(
        model_id=model_id,
        author=author,
        likes=likes,
        downloads=downloads,
        pipeline_tag=pipeline_tag,
        tags=tags if tags is not None else ["text-generation"],
        library_name=library_name,
        license=license_str,
        card_text=card_text,
        rank=rank,
        raw=raw or {"created_at": "2025-01-01"},
    )


def _make_search_result(
    model_id: str = "test/Model-7B",
    score: float = 0.85,
    bank_score: float = 0.9,
    anchor_score: float = 0.8,
    spread_score: float = 0.7,
    fuzzy_score: float = 0.6,
    positions: dict | None = None,
    anchor_labels: list[str] | None = None,
    vibe_summary: str = "A versatile model.",
    author: str = "test",
) -> SearchResult:
    return SearchResult(
        model_id=model_id,
        score=score,
        bank_score=bank_score,
        anchor_score=anchor_score,
        spread_score=spread_score,
        fuzzy_score=fuzzy_score,
        positions=positions or {},
        anchor_labels=anchor_labels or ["decoder-only", "chat"],
        vibe_summary=vibe_summary,
        author=author,
    )


class TestStructuredToDict:
    def test_all_fields_present(self):
        r = _make_structured()
        d = structured_to_dict(r)
        assert d["model_id"] == "test/Model-7B"
        assert d["author"] == "test"
        assert d["likes"] == 100
        assert d["downloads"] == 5000
        assert d["pipeline_tag"] == "text-generation"
        assert d["tags"] == ["text-generation"]
        assert d["library_name"] == "transformers"
        assert d["license"] == "apache-2.0"
        assert d["card_text"] == "A test model."
        assert d["rank"] == 0

    def test_empty_fields(self):
        r = _make_structured(
            author="", tags=[], library_name="", license_str="", card_text=""
        )
        d = structured_to_dict(r)
        assert d["author"] == ""
        assert d["tags"] == []
        assert d["library_name"] == ""


class TestCandidatesToDicts:
    def test_converts_list(self):
        candidates = [
            _make_structured(model_id="a/A", rank=0, raw={"created_at": "2025-01-01"}),
            _make_structured(model_id="b/B", rank=1, raw={"created_at": "2025-06-01"}),
        ]
        result = candidates_to_dicts(candidates)
        assert len(result) == 2
        assert result[0]["model_id"] == "a/A"
        assert result[1]["model_id"] == "b/B"
        assert result[0]["created_at"] == "2025-01-01"

    def test_empty_list(self):
        assert candidates_to_dicts([]) == []

    def test_includes_all_extract_fields(self):
        c = _make_structured(raw={"created_at": "2025-03-15"})
        result = candidates_to_dicts([c])
        d = result[0]
        assert "model_id" in d
        assert "author" in d
        assert "pipeline_tag" in d
        assert "tags" in d
        assert "library_name" in d
        assert "likes" in d
        assert "downloads" in d
        assert "created_at" in d
        assert "license" in d
        assert "card_text" in d


class TestFormatNetworkResults:
    def test_formats_search_results(self):
        results = [_make_search_result(), _make_search_result(model_id="b/B")]
        formatted = format_network_results(results)
        assert len(formatted) == 2
        assert formatted[0]["model_id"] == "test/Model-7B"
        assert formatted[0]["score"] == 0.85
        assert formatted[0]["score_breakdown"]["bank_proximity"] == 0.9
        assert formatted[0]["score_breakdown"]["anchor_overlap"] == 0.8
        assert formatted[0]["score_breakdown"]["spreading"] == 0.7
        assert formatted[0]["score_breakdown"]["fuzzy"] == 0.6

    def test_anchors_capped_at_15(self):
        labels = [f"anchor-{i}" for i in range(20)]
        r = _make_search_result(anchor_labels=labels)
        formatted = format_network_results([r])
        assert len(formatted[0]["anchors"]) == 15

    def test_includes_vibe_and_positions(self):
        positions = {"ARCHITECTURE": {"sign": 0, "depth": 0}}
        r = _make_search_result(positions=positions, vibe_summary="Great for code.")
        formatted = format_network_results([r])
        assert formatted[0]["vibe"] == "Great for code."
        assert formatted[0]["positions"] == positions

    def test_empty_results(self):
        assert format_network_results([]) == []

    def test_scores_rounded_to_four_decimals(self):
        r = _make_search_result(
            score=0.123456789,
            bank_score=0.111111,
            anchor_score=0.222222,
            spread_score=0.333333,
            fuzzy_score=0.444444,
        )
        formatted = format_network_results([r])
        assert formatted[0]["score"] == 0.1235
        assert formatted[0]["score_breakdown"]["bank_proximity"] == 0.1111
        assert formatted[0]["score_breakdown"]["anchor_overlap"] == 0.2222
        assert formatted[0]["score_breakdown"]["spreading"] == 0.3333
        assert formatted[0]["score_breakdown"]["fuzzy"] == 0.4444


class TestFormatFuzzyResults:
    def test_sorts_by_score_descending(self):
        candidates = [
            {
                "model_id": "a/Low",
                "author": "a",
                "likes": 10,
                "downloads": 100,
                "pipeline_tag": "text-generation",
                "tags": ["t1"],
                "library_name": "transformers",
            },
            {
                "model_id": "b/High",
                "author": "b",
                "likes": 500,
                "downloads": 10000,
                "pipeline_tag": "text-generation",
                "tags": ["t1"],
                "library_name": "transformers",
            },
        ]
        fuzzy_scores = {"a/Low": 0.3, "b/High": 0.9}
        result = format_fuzzy_results(candidates, fuzzy_scores, limit=10)
        assert result[0]["model_id"] == "b/High"
        assert result[1]["model_id"] == "a/Low"

    def test_respects_limit(self):
        candidates = [
            {
                "model_id": f"t/{i}",
                "author": "t",
                "likes": 0,
                "downloads": 0,
                "pipeline_tag": "",
                "tags": [],
                "library_name": "",
            }
            for i in range(5)
        ]
        fuzzy_scores = {f"t/{i}": float(i) for i in range(5)}
        result = format_fuzzy_results(candidates, fuzzy_scores, limit=2)
        assert len(result) == 2
        # Highest scores first
        assert result[0]["model_id"] == "t/4"
        assert result[1]["model_id"] == "t/3"

    def test_missing_fuzzy_score_defaults_zero(self):
        candidates = [
            {
                "model_id": "a/A",
                "author": "a",
                "likes": 0,
                "downloads": 0,
                "pipeline_tag": "",
                "tags": [],
                "library_name": "",
            },
        ]
        result = format_fuzzy_results(candidates, {}, limit=10)
        assert len(result) == 1
        assert result[0]["score"] == 0.0

    def test_tags_capped_at_10(self):
        candidates = [
            {
                "model_id": "a/A",
                "author": "a",
                "likes": 0,
                "downloads": 0,
                "pipeline_tag": "",
                "tags": [f"tag{i}" for i in range(20)],
                "library_name": "",
            },
        ]
        result = format_fuzzy_results(candidates, {"a/A": 0.5}, limit=10)
        assert len(result[0]["tags"]) == 10

    def test_none_tags_handled(self):
        candidates = [
            {
                "model_id": "a/A",
                "author": "a",
                "likes": 0,
                "downloads": 0,
                "pipeline_tag": "",
                "tags": None,
                "library_name": "",
            },
        ]
        result = format_fuzzy_results(candidates, {"a/A": 0.5}, limit=10)
        assert result[0]["tags"] == []

    def test_empty_candidates(self):
        assert format_fuzzy_results([], {}, limit=10) == []

    def test_output_field_names(self):
        candidates = [
            {
                "model_id": "a/A",
                "author": "a",
                "likes": 42,
                "downloads": 999,
                "pipeline_tag": "text-gen",
                "tags": ["t"],
                "library_name": "torch",
            },
        ]
        result = format_fuzzy_results(candidates, {"a/A": 0.75}, limit=10)
        d = result[0]
        assert d["model_id"] == "a/A"
        assert d["author"] == "a"
        assert d["score"] == 0.75
        assert d["likes"] == 42
        assert d["downloads"] == 999
        assert d["pipeline_tag"] == "text-gen"
        assert d["library"] == "torch"
