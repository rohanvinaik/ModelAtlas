"""Tests for the navigational query engine."""

from __future__ import annotations

from model_atlas.query import compare, invalidate_idf_cache, lineage, search, similar_to
from model_atlas.query_types import StructuredQuery


class TestSearch:
    def test_basic_search(self, populated_conn):
        results = search(populated_conn, "language model", limit=10)
        assert len(results) > 0
        assert all(r.score >= 0 for r in results)

    def test_code_query_favors_code_model(self, populated_conn):
        results = search(populated_conn, "small code model", limit=10)
        ids = [r.model_id for r in results]
        # Qwen Coder should rank higher for a code query
        qwen_idx = (
            ids.index("Qwen/Qwen2.5-Coder-1.5B")
            if "Qwen/Qwen2.5-Coder-1.5B" in ids
            else 999
        )
        assert qwen_idx < len(results)

    def test_fuzzy_scores_incorporated(self, populated_conn):
        fuzzy = {"meta-llama/Llama-3.1-8B-Instruct": 0.95}
        results = search(populated_conn, "llama", limit=10, fuzzy_scores=fuzzy)
        # Llama should get a boost from fuzzy
        llama = next(
            (r for r in results if "Llama-3.1-8B-Instruct" in r.model_id), None
        )
        assert llama is not None
        assert llama.fuzzy_score == 0.95

    def test_empty_network(self, conn):
        results = search(conn, "anything", limit=10)
        assert results == []


class TestSimilarTo:
    def test_similar_finds_related(self, populated_conn):
        results = similar_to(
            populated_conn, "meta-llama/Llama-3.1-8B-Instruct", limit=10
        )
        ids = [r.model_id for r in results]
        # The GGUF variant shares the most anchors
        assert "TheBloke/Llama-3.1-8B-Instruct-GGUF" in ids

    def test_similar_excludes_self(self, populated_conn):
        results = similar_to(populated_conn, "meta-llama/Llama-3.1-8B-Instruct")
        ids = [r.model_id for r in results]
        assert "meta-llama/Llama-3.1-8B-Instruct" not in ids


class TestCompare:
    def test_compare_two_models(self, populated_conn):
        result = compare(
            populated_conn,
            ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-Coder-1.5B"],
        )
        assert len(result.models) == 2
        assert "decoder-only" in result.shared_anchors
        assert result.jaccard_similarity > 0
        assert result.jaccard_similarity < 1
        # Each should have unique anchors
        assert len(result.per_model_unique["meta-llama/Llama-3.1-8B-Instruct"]) > 0
        assert len(result.per_model_unique["Qwen/Qwen2.5-Coder-1.5B"]) > 0

    def test_compare_includes_bank_deltas(self, populated_conn):
        result = compare(
            populated_conn,
            ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-Coder-1.5B"],
        )
        assert "EFFICIENCY" in result.bank_deltas
        eff = result.bank_deltas["EFFICIENCY"]
        # Llama is 7B (0), Qwen is 1.5B (-2)
        assert eff["meta-llama/Llama-3.1-8B-Instruct"] == 0
        assert eff["Qwen/Qwen2.5-Coder-1.5B"] == -2


class TestLineage:
    def test_lineage_with_links(self, populated_conn):
        result = lineage(populated_conn, "TheBloke/Llama-3.1-8B-Instruct-GGUF")
        assert result["model_id"] == "TheBloke/Llama-3.1-8B-Instruct-GGUF"
        assert len(result["derived_from"]) == 1
        assert (
            result["derived_from"][0]["target_id"] == "meta-llama/Llama-3.1-8B-Instruct"
        )

    def test_lineage_not_found(self, conn):
        result = lineage(conn, "nonexistent/Model")
        assert "error" in result


class TestInvalidateIdfCache:
    def test_clears_cache(self, populated_conn):
        """invalidate_idf_cache resets the module-level IDF cache."""
        # Run a search to populate the IDF cache
        search(populated_conn, "model", limit=1)
        # Clear it
        invalidate_idf_cache()
        # Should not raise — cache is just empty now
        results = search(populated_conn, "model", limit=1)
        assert len(results) >= 0

    def test_idempotent(self):
        """Calling invalidate_idf_cache twice doesn't raise."""
        invalidate_idf_cache()
        invalidate_idf_cache()


class TestStructuredQueryBankDirections:
    def test_returns_only_set_banks(self):
        sq = StructuredQuery(efficiency=-1, capability=1)
        dirs = sq.bank_directions()
        assert dirs == {"EFFICIENCY": -1, "CAPABILITY": 1}
        assert "ARCHITECTURE" not in dirs

    def test_all_banks_set(self):
        sq = StructuredQuery(
            architecture=0, capability=1, efficiency=-1,
            compatibility=1, lineage=0, domain=1, quality=-1,
        )
        dirs = sq.bank_directions()
        assert len(dirs) == 7

    def test_no_banks_set(self):
        sq = StructuredQuery()
        assert sq.bank_directions() == {}

    def test_zero_is_included(self):
        """Direction 0 (near-zero) is a valid direction, not omitted."""
        sq = StructuredQuery(architecture=0)
        dirs = sq.bank_directions()
        assert dirs == {"ARCHITECTURE": 0}
