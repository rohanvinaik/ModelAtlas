"""Tests for fuzzy string matching search layer."""

from __future__ import annotations

from model_atlas.search.fuzzy import (
    STOP_WORDS,
    FuzzyScore,
    _build_searchable_strings,
    _tokenize_query,
    score_models,
)


class TestTokenizeQuery:
    def test_basic_tokenization(self):
        tokens = _tokenize_query("llama instruct model")
        assert "llama" in tokens
        assert "instruct" in tokens

    def test_stop_words_removed(self):
        tokens = _tokenize_query("the best model for code generation")
        assert "the" not in tokens
        assert "for" not in tokens
        assert "model" not in tokens  # "model" is in STOP_WORDS

    def test_single_char_tokens_removed(self):
        tokens = _tokenize_query("a b c llama")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "llama" in tokens

    def test_case_insensitive(self):
        tokens = _tokenize_query("Llama INSTRUCT Code")
        assert "llama" in tokens
        assert "instruct" in tokens
        assert "code" in tokens

    def test_special_characters_split(self):
        tokens = _tokenize_query("meta-llama/Llama-3.1-8B")
        # Hyphens and dots are preserved in tokens by the regex
        assert "meta-llama" in tokens
        assert "llama-3.1-8b" in tokens

    def test_empty_query(self):
        tokens = _tokenize_query("")
        assert tokens == []

    def test_only_stop_words(self):
        tokens = _tokenize_query("the a an is are model")
        assert tokens == []

    def test_numbers_preserved(self):
        tokens = _tokenize_query("7B 70B 128k")
        assert "7b" in tokens
        assert "70b" in tokens
        assert "128k" in tokens

    def test_underscores_preserved_in_tokens(self):
        tokens = _tokenize_query("text_generation chat_template")
        assert "text_generation" in tokens
        assert "chat_template" in tokens

    def test_dots_preserved_in_tokens(self):
        tokens = _tokenize_query("llama-3.1 qwen2.5")
        # Hyphens and dots stay within tokens
        assert "llama-3.1" in tokens
        assert "qwen2.5" in tokens


class TestBuildSearchableStrings:
    def test_model_id_split_into_parts(self):
        result = _build_searchable_strings(
            "meta-llama/Llama-3-8B", ["text-generation"], "text-generation", ""
        )
        assert "meta" in result["model_id"]
        assert "llama" in result["model_id"]
        assert "8b" in result["model_id"]

    def test_tags_joined(self):
        result = _build_searchable_strings(
            "test/Model", ["code", "instruct", "transformers"], "", ""
        )
        assert "code" in result["tags"]
        assert "instruct" in result["tags"]

    def test_pipeline_tag_lowered(self):
        result = _build_searchable_strings(
            "test/Model", [], "Text-Generation", ""
        )
        assert result["pipeline_tag"] == "text-generation"

    def test_card_text_truncated(self):
        long_text = "x" * 1000
        result = _build_searchable_strings("test/Model", [], "", long_text)
        assert len(result["card_text"]) == 500

    def test_empty_fields(self):
        result = _build_searchable_strings("test/Model", [], "", "")
        assert result["tags"] == ""
        assert result["pipeline_tag"] == ""
        assert result["card_text"] == ""

    def test_all_fields_are_lowercase(self):
        result = _build_searchable_strings(
            "Meta/BIG-MODEL",
            ["CODE", "INSTRUCT"],
            "Text-Generation",
            "Some Card Text",
        )
        for key, value in result.items():
            assert value == value.lower(), f"Field {key} is not lowercased"


class TestScoreModels:
    def _make_model(self, model_id, tags=None, pipeline_tag="", card_text=""):
        return {
            "model_id": model_id,
            "tags": tags or [],
            "pipeline_tag": pipeline_tag,
            "card_text": card_text,
        }

    def test_exact_name_match_scores_high(self):
        models = [
            self._make_model("meta-llama/Llama-3.1-8B-Instruct"),
            self._make_model("mistralai/Mistral-7B-v0.1"),
        ]
        scores = score_models("llama instruct", models)
        llama_score = next(s for s in scores if "Llama" in s.model_id)
        mistral_score = next(s for s in scores if "Mistral" in s.model_id)
        assert llama_score.score > mistral_score.score

    def test_tag_matching(self):
        models = [
            self._make_model("test/ModelA", tags=["code", "python"]),
            self._make_model("test/ModelB", tags=["medical", "clinical"]),
        ]
        scores = score_models("python code", models)
        code_score = next(s for s in scores if s.model_id == "test/ModelA")
        med_score = next(s for s in scores if s.model_id == "test/ModelB")
        assert code_score.score > med_score.score

    def test_empty_query_returns_zero_scores(self):
        models = [
            self._make_model("test/Model"),
        ]
        scores = score_models("", models)
        assert len(scores) == 1
        assert scores[0].score == 0.0

    def test_stop_word_only_query_returns_zero_scores(self):
        models = [
            self._make_model("test/Model"),
        ]
        scores = score_models("the a an is model", models)
        assert len(scores) == 1
        assert scores[0].score == 0.0

    def test_returns_fuzzy_score_objects(self):
        models = [self._make_model("test/Model")]
        scores = score_models("test", models)
        assert len(scores) == 1
        assert isinstance(scores[0], FuzzyScore)
        assert isinstance(scores[0].score, float)
        assert isinstance(scores[0].model_id, str)
        assert isinstance(scores[0].best_match_field, str)
        assert isinstance(scores[0].best_match_value, str)

    def test_scores_between_zero_and_one(self):
        models = [
            self._make_model("meta-llama/Llama-3.1-8B"),
            self._make_model("test/random-model"),
        ]
        scores = score_models("llama 8b", models)
        for s in scores:
            assert 0.0 <= s.score <= 1.0

    def test_empty_model_list(self):
        scores = score_models("llama", [])
        assert scores == []

    def test_card_text_matching(self):
        models = [
            self._make_model(
                "test/ModelA",
                card_text="This model excels at code generation and Python programming.",
            ),
            self._make_model(
                "test/ModelB",
                card_text="A model for medical image classification.",
            ),
        ]
        scores = score_models("python programming", models)
        code_score = next(s for s in scores if s.model_id == "test/ModelA")
        med_score = next(s for s in scores if s.model_id == "test/ModelB")
        assert code_score.score > med_score.score

    def test_best_match_field_populated(self):
        models = [
            self._make_model(
                "generic/Model",
                tags=["llama", "instruct"],
            ),
        ]
        scores = score_models("llama", models)
        assert scores[0].best_match_field != ""
        assert scores[0].best_match_value != ""

    def test_multiple_tokens_averaged(self):
        """Score should be an average across all query tokens."""
        models = [
            self._make_model("meta-llama/Llama-3.1-8B"),
        ]
        # "llama" should match well, "zebra" should not
        scores = score_models("llama zebra", models)
        # The average should be lower than a pure "llama" match
        pure_scores = score_models("llama", models)
        assert scores[0].score < pure_scores[0].score

    def test_pipeline_tag_matching(self):
        models = [
            self._make_model("test/GenModel", pipeline_tag="text-generation"),
            self._make_model("test/ClassModel", pipeline_tag="text-classification"),
        ]
        scores = score_models("generation", models)
        gen_score = next(s for s in scores if s.model_id == "test/GenModel")
        cls_score = next(s for s in scores if s.model_id == "test/ClassModel")
        assert gen_score.score > cls_score.score


class TestStopWords:
    def test_common_stop_words_present(self):
        for word in ["the", "a", "an", "is", "are", "for", "and", "or"]:
            assert word in STOP_WORDS

    def test_model_is_stop_word(self):
        """The word 'model' is too generic in this context."""
        assert "model" in STOP_WORDS
        assert "models" in STOP_WORDS

    def test_stop_words_are_lowercase(self):
        for word in STOP_WORDS:
            assert word == word.lower()
