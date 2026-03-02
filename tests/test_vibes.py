"""Tests for Tier 3 vibe extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

from model_atlas.extraction.vibes import (
    VibeExtractor,
    VibeOutput,
    build_vibe_prompt,
    extract_vibe_summary,
)


class TestVibeOutput:
    def test_dataclass_fields(self):
        v = VibeOutput(summary="A great model", extra_anchors=["fast", "small"])
        assert v.summary == "A great model"
        assert v.extra_anchors == ["fast", "small"]

    def test_empty_anchors(self):
        v = VibeOutput(summary="Minimal model", extra_anchors=[])
        assert v.extra_anchors == []


class TestBuildVibePrompt:
    def test_basic_prompt(self):
        prompt = build_vibe_prompt(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            author="meta-llama",
            pipeline_tag="text-generation",
            tags=["text-generation", "llama"],
            param_count="8B parameters",
            family="Llama-family",
            capabilities=["instruction-following", "chat"],
        )
        assert "meta-llama/Llama-3.1-8B-Instruct" in prompt
        assert "meta-llama" in prompt
        assert "text-generation" in prompt
        assert "8B parameters" in prompt
        assert "Llama-family" in prompt
        assert "instruction-following" in prompt

    def test_prompt_defaults(self):
        prompt = build_vibe_prompt(model_id="test/Model")
        assert "test/Model" in prompt
        assert "unknown" in prompt  # defaults

    def test_tags_truncated_to_15(self):
        tags = [f"tag-{i}" for i in range(30)]
        prompt = build_vibe_prompt(model_id="test/Model", tags=tags)
        assert "tag-14" in prompt
        assert "tag-15" not in prompt

    def test_no_tags(self):
        prompt = build_vibe_prompt(model_id="test/Model", tags=[])
        assert "none" in prompt


class TestVibeExtractor:
    def test_not_loaded_raises(self):
        extractor = VibeExtractor(model_name="test/model")
        assert not extractor.is_loaded
        try:
            extractor.extract("test prompt")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not loaded" in str(e)

    def test_load_and_extract(self):
        """Test that extract() returns a VibeOutput from model output."""
        mock_generator = MagicMock()
        mock_generator.return_value = {
            "summary": "A versatile code model",
            "extra_anchors": ["multi-language", "fast-inference"],
        }

        extractor = VibeExtractor(model_name="test/model")
        # Directly inject the mock generator to skip real model loading
        extractor._generator = mock_generator

        assert extractor.is_loaded

        result = extractor.extract("test prompt")
        assert isinstance(result, VibeOutput)
        assert result.summary == "A versatile code model"
        assert "multi-language" in result.extra_anchors
        mock_generator.assert_called_once_with("test prompt")

    def test_extract_truncates_anchors(self):
        """Extra anchors capped at 5."""
        extractor = VibeExtractor(model_name="test/model")
        mock_gen = MagicMock()
        mock_gen.return_value = {
            "summary": "test",
            "extra_anchors": ["a", "b", "c", "d", "e", "f", "g"],
        }
        extractor._generator = mock_gen

        result = extractor.extract("prompt")
        assert len(result.extra_anchors) <= 5


class TestBackwardCompat:
    def test_extract_vibe_summary_returns_empty(self):
        """The old stub still returns empty for pipeline compatibility."""
        assert extract_vibe_summary("test/Model") == ""
        assert extract_vibe_summary("test/Model", card_text="some text") == ""
        assert extract_vibe_summary(
            "test/Model", pipeline_tag="text-gen", author="test"
        ) == ""
