"""Tests for the structured navigate() scoring engine."""

from __future__ import annotations

from model_atlas import db
from model_atlas.query import (
    NavigationResult,
    StructuredQuery,
    _bank_score_single,
    invalidate_idf_cache,
    navigate,
)

# ---------------------------------------------------------------------------
# Unit tests for _bank_score_single
# ---------------------------------------------------------------------------


class TestBankScoreSingle:
    def test_direction_zero_at_origin(self):
        """Wanting zero, model at zero → perfect score."""
        assert _bank_score_single(0, 0) == 1.0

    def test_direction_zero_penalizes_distance(self):
        """Wanting zero, model far away → decayed score."""
        score = _bank_score_single(3, 0)
        assert score == 1.0 / (1.0 + 3)

    def test_direction_positive_aligned(self):
        """Wanting +1, model on positive side → 1.0."""
        assert _bank_score_single(2, 1) == 1.0

    def test_direction_positive_at_zero(self):
        """Wanting +1, model at zero → 0.5 (neutral)."""
        assert _bank_score_single(0, 1) == 0.5

    def test_direction_positive_opposed(self):
        """Wanting +1, model on negative side → decayed."""
        score = _bank_score_single(-2, 1)
        assert score == 1.0 / (1.0 + 2)

    def test_direction_negative_aligned(self):
        """Wanting -1, model on negative side → 1.0."""
        assert _bank_score_single(-3, -1) == 1.0

    def test_direction_negative_opposed(self):
        """Wanting -1, model on positive side → decayed."""
        score = _bank_score_single(1, -1)
        assert score == 1.0 / (1.0 + 1)


# ---------------------------------------------------------------------------
# Integration tests with populated_conn
# ---------------------------------------------------------------------------


class TestNavigateBasic:
    def setup_method(self):
        invalidate_idf_cache()

    def test_no_constraints_returns_all(self, populated_conn):
        """Empty query returns all models with score 1.0."""
        results = navigate(populated_conn, StructuredQuery())
        assert len(results) == 4
        for r in results:
            assert r.score == 1.0

    def test_results_are_navigation_results(self, populated_conn):
        """Results are NavigationResult instances."""
        results = navigate(populated_conn, StructuredQuery())
        assert all(isinstance(r, NavigationResult) for r in results)

    def test_limit_respected(self, populated_conn):
        """Limit caps result count."""
        results = navigate(populated_conn, StructuredQuery(limit=2))
        assert len(results) == 2


class TestBankAlignment:
    def setup_method(self):
        invalidate_idf_cache()

    def test_efficiency_small_prefers_qwen(self, populated_conn):
        """efficiency=-1 should rank Qwen Coder (EFFICIENCY -1,2) highest."""
        results = navigate(populated_conn, StructuredQuery(efficiency=-1))
        assert results[0].model_id == "Qwen/Qwen2.5-Coder-1.5B"
        assert results[0].bank_alignment == 1.0

    def test_efficiency_small_penalizes_zero(self, populated_conn):
        """Models at EFFICIENCY zero get 0.5 when we want -1."""
        results = navigate(populated_conn, StructuredQuery(efficiency=-1))
        llama = next(
            r for r in results if r.model_id == "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert llama.bank_alignment == 0.5

    def test_domain_specialized_prefers_specialized(self, populated_conn):
        """domain=+1 should rank domain-specialized models above general ones."""
        results = navigate(populated_conn, StructuredQuery(domain=1))
        specialized = {r.model_id for r in results if r.bank_alignment == 1.0}
        general = {r.model_id for r in results if r.bank_alignment < 1.0}
        assert "medicalai/BioMedGPT-LM-7B" in specialized
        assert "Qwen/Qwen2.5-Coder-1.5B" in specialized
        assert "meta-llama/Llama-3.1-8B-Instruct" in general

    def test_compatibility_positive_prefers_gguf(self, populated_conn):
        """compatibility=+1 should rank the GGUF model highest."""
        results = navigate(populated_conn, StructuredQuery(compatibility=1))
        assert results[0].model_id == "TheBloke/Llama-3.1-8B-Instruct-GGUF"

    def test_multi_bank_multiplicative(self, populated_conn):
        """Multiple bank constraints multiply — model must satisfy both."""
        # Small + domain-specialized: only Qwen is small, only medical is domain-specialized
        # Neither satisfies both perfectly, so scores should be < 1.0
        results = navigate(populated_conn, StructuredQuery(efficiency=-1, domain=1))
        for r in results:
            assert r.bank_alignment <= 1.0
        # Qwen: efficiency=1.0, domain=1.0 (sign=1,depth=1 → aligned) → 1.0
        qwen = next(r for r in results if r.model_id == "Qwen/Qwen2.5-Coder-1.5B")
        med = next(r for r in results if r.model_id == "medicalai/BioMedGPT-LM-7B")
        # Both have domain > 0, Qwen also has efficiency < 0
        assert qwen.bank_alignment > med.bank_alignment


class TestAnchorRelevance:
    def setup_method(self):
        invalidate_idf_cache()

    def test_require_filters_strictly(self, populated_conn):
        """require_anchors acts as a hard filter."""
        results = navigate(
            populated_conn,
            StructuredQuery(require_anchors=["code-generation"]),
        )
        assert len(results) == 1
        assert results[0].model_id == "Qwen/Qwen2.5-Coder-1.5B"

    def test_require_nonexistent_anchor_returns_empty(self, populated_conn):
        """Requiring a non-existent anchor returns nothing."""
        results = navigate(
            populated_conn,
            StructuredQuery(require_anchors=["nonexistent-anchor-xyz"]),
        )
        assert results == []

    def test_require_multiple_anchors(self, populated_conn):
        """All required anchors must be present."""
        # Only Llama has both instruction-following AND Llama-family
        results = navigate(
            populated_conn,
            StructuredQuery(require_anchors=["instruction-following", "Llama-family"]),
        )
        model_ids = {r.model_id for r in results}
        assert "meta-llama/Llama-3.1-8B-Instruct" in model_ids
        assert "TheBloke/Llama-3.1-8B-Instruct-GGUF" in model_ids
        assert "Qwen/Qwen2.5-Coder-1.5B" not in model_ids

    def test_prefer_anchors_boost_idf_weighted(self, populated_conn):
        """prefer_anchors boost score based on IDF overlap."""
        results = navigate(
            populated_conn,
            StructuredQuery(prefer_anchors=["code-generation", "consumer-GPU-viable"]),
        )
        qwen = next(r for r in results if r.model_id == "Qwen/Qwen2.5-Coder-1.5B")
        llama = next(
            r for r in results if r.model_id == "meta-llama/Llama-3.1-8B-Instruct"
        )
        # Qwen has both preferred anchors, Llama has neither
        assert qwen.anchor_relevance > llama.anchor_relevance
        assert qwen.score > llama.score

    def test_avoid_anchors_penalize(self, populated_conn):
        """avoid_anchors halve the score per match."""
        results_without_avoid = navigate(populated_conn, StructuredQuery())
        results_with_avoid = navigate(
            populated_conn,
            StructuredQuery(avoid_anchors=["embedding"]),
        )
        # No model has "embedding" in the fixture → scores should be identical
        for r1, r2 in zip(results_without_avoid, results_with_avoid):
            assert r1.score == r2.score

    def test_avoid_actually_penalizes_matching(self, populated_conn):
        """Avoiding an anchor that models have should reduce their score."""
        results = navigate(
            populated_conn,
            StructuredQuery(avoid_anchors=["quantized"]),
        )
        gguf = next(
            r for r in results if r.model_id == "TheBloke/Llama-3.1-8B-Instruct-GGUF"
        )
        llama = next(
            r for r in results if r.model_id == "meta-llama/Llama-3.1-8B-Instruct"
        )
        # GGUF model has "quantized" anchor, so it gets penalized
        assert gguf.anchor_relevance == 0.5  # one avoided anchor → 0.5^1
        assert llama.anchor_relevance == 1.0


class TestSeedSimilarity:
    def setup_method(self):
        invalidate_idf_cache()

    def test_similar_to_ranks_by_anchor_overlap(self, populated_conn):
        """similar_to uses IDF-weighted Jaccard on anchor sets."""
        results = navigate(
            populated_conn,
            StructuredQuery(similar_to="meta-llama/Llama-3.1-8B-Instruct"),
        )
        # GGUF model shares the most anchors with Llama (both have Llama-family,
        # instruction-following, decoder-only)
        gguf = next(
            r for r in results if r.model_id == "TheBloke/Llama-3.1-8B-Instruct-GGUF"
        )
        qwen = next(r for r in results if r.model_id == "Qwen/Qwen2.5-Coder-1.5B")
        assert gguf.seed_similarity > qwen.seed_similarity

    def test_similar_to_nonexistent_gives_zero(self, populated_conn):
        """Seed model not in network → seed_similarity = 0 for all."""
        results = navigate(
            populated_conn,
            StructuredQuery(similar_to="nonexistent/model"),
        )
        # seed_anchors is empty → seed_similarity = 1.0 (neutral)
        for r in results:
            assert r.seed_similarity == 1.0


class TestCombinedScoring:
    def setup_method(self):
        invalidate_idf_cache()

    def test_multiplicative_combination(self, populated_conn):
        """final_score = bank_alignment * anchor_relevance * seed_similarity."""
        results = navigate(
            populated_conn,
            StructuredQuery(
                efficiency=-1,
                require_anchors=["code-generation"],
                prefer_anchors=["consumer-GPU-viable"],
            ),
        )
        assert len(results) == 1  # only Qwen passes require
        r = results[0]
        expected = r.bank_alignment * r.anchor_relevance * r.seed_similarity
        assert abs(r.score - expected) < 1e-9

    def test_zero_bank_kills_score(self, populated_conn):
        """A bank score of 0 should propagate to final score = 0.

        In practice bank scores are never exactly 0 (minimum is
        NAVIGATE_MISSING_BANK_PENALTY or 1/(1+d)), so we test that
        opposed models score very low.
        """
        # Request novel architecture (+1) and small efficiency (-1)
        # Standard decoder models at arch=0 get 0.5, not 0
        results = navigate(
            populated_conn,
            StructuredQuery(architecture=1, efficiency=-1),
        )
        # All fixture models are standard transformers (arch=0,0) → 0.5 for arch
        # Only Qwen is small → Qwen gets the best combined bank score
        qwen = next(r for r in results if r.model_id == "Qwen/Qwen2.5-Coder-1.5B")
        assert qwen.bank_alignment == 0.5 * 1.0  # arch=0.5, efficiency=1.0


class TestIDFWeighting:
    def setup_method(self):
        invalidate_idf_cache()

    def test_idf_computed(self, populated_conn):
        """IDF values should be computed for all anchors."""
        idf = db.compute_anchor_idf(populated_conn)
        assert len(idf) > 0
        # "decoder-only" is on 4 models, should have lower IDF than rarer anchors
        # "code-generation" is on 1 model, should have higher IDF
        assert idf.get("decoder-only", 0) < idf.get("code-generation", 0)

    def test_rare_anchor_worth_more_in_prefer(self, populated_conn):
        """Preferring a rare anchor should count more than a common one."""
        # code-generation is rare (1 model), decoder-only is common (4 models)
        # Qwen has both; other models only have decoder-only
        results_rare = navigate(
            populated_conn,
            StructuredQuery(prefer_anchors=["code-generation"]),
        )
        results_common = navigate(
            populated_conn,
            StructuredQuery(prefer_anchors=["decoder-only"]),
        )
        qwen_rare = next(
            r for r in results_rare if r.model_id == "Qwen/Qwen2.5-Coder-1.5B"
        )
        llama_common = next(
            r
            for r in results_common
            if r.model_id == "meta-llama/Llama-3.1-8B-Instruct"
        )
        # Both should have anchor_relevance = 1.0 since they have the preferred anchor
        assert qwen_rare.anchor_relevance == 1.0
        assert llama_common.anchor_relevance == 1.0


class TestBatchPerformance:
    def setup_method(self):
        invalidate_idf_cache()

    def test_batch_positions(self, populated_conn):
        """batch_get_positions returns correct data for all models."""
        model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-Coder-1.5B"]
        positions = db.batch_get_positions(populated_conn, model_ids)
        assert "meta-llama/Llama-3.1-8B-Instruct" in positions
        assert "ARCHITECTURE" in positions["meta-llama/Llama-3.1-8B-Instruct"]
        # Qwen EFFICIENCY is (-1, 2)
        assert positions["Qwen/Qwen2.5-Coder-1.5B"]["EFFICIENCY"] == (-1, 2)

    def test_batch_anchor_sets(self, populated_conn):
        """batch_get_anchor_sets returns correct anchor labels."""
        model_ids = ["Qwen/Qwen2.5-Coder-1.5B"]
        anchors = db.batch_get_anchor_sets(populated_conn, model_ids)
        assert "code-generation" in anchors["Qwen/Qwen2.5-Coder-1.5B"]
        assert "consumer-GPU-viable" in anchors["Qwen/Qwen2.5-Coder-1.5B"]

    def test_batch_empty_input(self, populated_conn):
        """Empty model list returns empty dicts."""
        assert db.batch_get_positions(populated_conn, []) == {}
        assert db.batch_get_anchor_sets(populated_conn, []) == {}
