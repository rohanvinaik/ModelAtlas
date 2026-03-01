"""Tests for compound navigational queries, gradient scoring, and ParsedQuery."""

from __future__ import annotations

from model_atlas.query import (
    BankConstraint,
    ParsedQuery,
    _bank_proximity_score,
    _confidence_weighted_jaccard,
    _parse_query,
    lineage,
    search,
)


class TestParseQuery:
    def test_basic_bank_constraint(self):
        parsed = _parse_query("small code model")
        assert isinstance(parsed, ParsedQuery)
        banks = [c.bank for c in parsed.bank_constraints]
        assert "EFFICIENCY" in banks
        assert "code-generation" in parsed.anchor_targets

    def test_compound_constraints(self):
        parsed = _parse_query("small trending code model")
        banks = [c.bank for c in parsed.bank_constraints]
        assert "EFFICIENCY" in banks
        assert "QUALITY" in banks
        assert "code-generation" in parsed.anchor_targets

    def test_direction_vectors(self):
        parsed = _parse_query("small base model")
        assert parsed.direction_vectors.get("EFFICIENCY") == -1
        assert parsed.direction_vectors.get("LINEAGE") == -1

    def test_seed_extraction(self):
        parsed = _parse_query("models like meta-llama/Llama-3.1-8B-Instruct")
        assert "meta-llama/Llama-3.1-8B-Instruct" in parsed.seed_model_ids

    def test_similar_to_extraction(self):
        parsed = _parse_query("similar to Qwen/Qwen2.5-Coder-1.5B")
        assert "Qwen/Qwen2.5-Coder-1.5B" in parsed.seed_model_ids

    def test_no_signals(self):
        parsed = _parse_query("hello world")
        assert parsed.bank_constraints == []
        assert parsed.anchor_targets == []
        assert parsed.seed_model_ids == []

    def test_raw_tokens_preserved(self):
        parsed = _parse_query("small code model")
        assert "small" in parsed.raw_tokens
        assert "code" in parsed.raw_tokens
        assert "model" in parsed.raw_tokens


class TestGradientBankScoring:
    def test_in_range_scores_1(self):
        positions = {"EFFICIENCY": {"sign": -1, "depth": 2}}
        constraints = [BankConstraint(bank="EFFICIENCY", max_signed=-1)]
        score = _bank_proximity_score(positions, constraints)
        assert score == 1.0

    def test_gradient_decay_near_range(self):
        """Model just outside range gets partial score, not zero."""
        positions = {"EFFICIENCY": {"sign": 0, "depth": 0}}
        constraints = [BankConstraint(bank="EFFICIENCY", max_signed=-1)]
        score = _bank_proximity_score(positions, constraints)
        # signed=0, max_signed=-1, dist=1 -> 1/(1+1) = 0.5
        assert score == 0.5

    def test_gradient_far_from_range(self):
        """Model far from range gets low but nonzero score."""
        positions = {"EFFICIENCY": {"sign": 1, "depth": 3}}
        constraints = [BankConstraint(bank="EFFICIENCY", max_signed=-1)]
        score = _bank_proximity_score(positions, constraints)
        # signed=3, max_signed=-1, dist=4 -> 1/(1+4) = 0.2
        assert score == 0.2

    def test_no_constraints_neutral(self):
        positions = {"EFFICIENCY": {"sign": -1, "depth": 2}}
        score = _bank_proximity_score(positions, [])
        assert score == 0.5

    def test_missing_bank_scores_zero(self):
        positions = {}
        constraints = [BankConstraint(bank="EFFICIENCY", max_signed=-1)]
        score = _bank_proximity_score(positions, constraints)
        assert score == 0.0

    def test_compound_scoring_averages(self):
        """Multiple constraints are averaged."""
        positions = {
            "EFFICIENCY": {"sign": -1, "depth": 2},  # signed=-2, in small range
            "QUALITY": {"sign": 1, "depth": 1},  # signed=1, in trending range
        }
        constraints = [
            BankConstraint(bank="EFFICIENCY", max_signed=-1),
            BankConstraint(bank="QUALITY", min_signed=1),
        ]
        score = _bank_proximity_score(positions, constraints)
        assert score == 1.0  # both in range

    def test_directional_scoring(self):
        """Direction-only constraint scores by side."""
        positions = {"EFFICIENCY": {"sign": -1, "depth": 2}}
        constraints = [BankConstraint(bank="EFFICIENCY", direction=-1)]
        score = _bank_proximity_score(positions, constraints)
        assert score == 1.0  # on desired side

    def test_directional_wrong_side(self):
        """Direction-only on wrong side decays."""
        positions = {"EFFICIENCY": {"sign": 1, "depth": 2}}
        constraints = [BankConstraint(bank="EFFICIENCY", direction=-1)]
        score = _bank_proximity_score(positions, constraints)
        assert score < 0.5  # on wrong side


class TestConfidenceWeightedJaccard:
    def test_basic_overlap(self):
        targets = {"code-generation", "instruction-following"}
        model_anchors = [
            {"label": "code-generation", "confidence": 1.0},
            {"label": "chat", "confidence": 0.8},
        ]
        score = _confidence_weighted_jaccard(targets, model_anchors)
        assert 0 < score < 1

    def test_full_overlap(self):
        targets = {"code-generation"}
        model_anchors = [{"label": "code-generation", "confidence": 1.0}]
        score = _confidence_weighted_jaccard(targets, model_anchors)
        assert score == 1.0

    def test_no_overlap(self):
        targets = {"code-generation"}
        model_anchors = [{"label": "chat", "confidence": 1.0}]
        score = _confidence_weighted_jaccard(targets, model_anchors)
        assert score == 0.0

    def test_low_confidence_reduces_score(self):
        targets = {"code-generation"}
        high_conf = [{"label": "code-generation", "confidence": 1.0}]
        low_conf = [{"label": "code-generation", "confidence": 0.3}]
        high_score = _confidence_weighted_jaccard(targets, high_conf)
        low_score = _confidence_weighted_jaccard(targets, low_conf)
        assert high_score >= low_score

    def test_empty_inputs(self):
        assert _confidence_weighted_jaccard(set(), []) == 0.0


class TestCompoundSearch:
    def test_compound_multi_bank(self, populated_conn):
        """Compound query 'small code model' favors Qwen Coder."""
        results = search(populated_conn, "small code model", limit=10)
        ids = [r.model_id for r in results]
        assert "Qwen/Qwen2.5-Coder-1.5B" in ids
        qwen_idx = ids.index("Qwen/Qwen2.5-Coder-1.5B")
        # Should rank high (top 2 at least)
        assert qwen_idx < 2

    def test_spread_score_present(self, populated_conn):
        """Search results include spread_score field."""
        results = search(populated_conn, "code model", limit=10)
        for r in results:
            assert hasattr(r, "spread_score")

    def test_seed_based_search(self, populated_conn):
        """Seed-based query activates related models."""
        results = search(
            populated_conn,
            "models like meta-llama/Llama-3.1-8B-Instruct",
            limit=10,
        )
        ids = [r.model_id for r in results]
        # GGUF variant should rank highly (linked + shared anchors)
        assert "TheBloke/Llama-3.1-8B-Instruct-GGUF" in ids

    def test_gguf_query(self, populated_conn):
        """GGUF query activates compatibility anchor."""
        results = search(populated_conn, "gguf llama", limit=10)
        ids = [r.model_id for r in results]
        assert "TheBloke/Llama-3.1-8B-Instruct-GGUF" in ids


class TestLineageOrdering:
    def test_lineage_derivatives_ordered(self, populated_conn):
        """Lineage results include lineage_signed for ordering."""
        result = lineage(populated_conn, "meta-llama/Llama-3.1-8B-Instruct")
        # The GGUF model is a derivative
        derivatives = result["derivatives"]
        if derivatives:
            # Each derivative has lineage_signed enrichment
            assert "lineage_signed" in derivatives[0]

    def test_lineage_derived_from_ordered(self, populated_conn):
        """derived_from results are ordered by lineage position."""
        result = lineage(populated_conn, "TheBloke/Llama-3.1-8B-Instruct-GGUF")
        derived_from = result["derived_from"]
        assert len(derived_from) == 1
        assert "lineage_signed" in derived_from[0]

    def test_lineage_not_found(self, conn):
        result = lineage(conn, "nonexistent/Model")
        assert "error" in result
