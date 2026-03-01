"""Tests for source adapters."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from model_atlas.sources.base import SourceAdapter, SourceSearchResult
from model_atlas.sources.huggingface import HuggingFaceAdapter
from model_atlas.sources.ollama import OllamaAdapter, _parse_param_size


class TestRegistry:
    def test_list_sources(self):
        from model_atlas.sources.registry import list_sources

        sources = list_sources()
        assert "huggingface" in sources
        # ollama may or may not be registered depending on environment

    def test_get_source_huggingface(self):
        from model_atlas.sources.registry import get_source

        adapter = get_source("huggingface")
        assert adapter.name == "huggingface"

    def test_get_source_unknown_raises(self):
        from model_atlas.sources.registry import get_source

        with pytest.raises(KeyError):
            get_source("nonexistent_source")

    def test_register_custom_source(self):
        from model_atlas.sources.registry import get_source, register_source

        class DummyAdapter(SourceAdapter):
            @property
            def name(self) -> str:
                return "dummy"

            def search(self, query, *, limit=20, filters=None):
                return []

            def get_detail(self, model_id):
                from model_atlas.extraction.deterministic import ModelInput

                return ModelInput(model_id=model_id)

        register_source(DummyAdapter())
        assert get_source("dummy").name == "dummy"


class TestHuggingFaceAdapter:
    def test_name(self):
        adapter = HuggingFaceAdapter()
        assert adapter.name == "huggingface"

    def test_search_returns_source_search_results(self):
        adapter = HuggingFaceAdapter()

        mock_model = MagicMock()
        mock_model.id = "test-user/test-model"
        mock_model.author = "test-user"
        mock_model.downloads = 1000
        mock_model.likes = 50
        mock_model.tags = ["text-generation", "pytorch"]
        mock_model.last_modified = None
        mock_model.sha = "abc123"

        with patch.object(adapter._api, "list_models", return_value=[mock_model]):
            results = adapter.search("test", limit=5)

        assert len(results) == 1
        assert isinstance(results[0], SourceSearchResult)
        assert results[0].model_id == "test-user/test-model"
        assert results[0].source == "huggingface"
        assert results[0].likes == 50

    def test_search_handles_api_error(self):
        adapter = HuggingFaceAdapter()
        with patch.object(
            adapter._api, "list_models", side_effect=Exception("API down")
        ):
            results = adapter.search("test")
        assert results == []


class TestOllamaAdapter:
    def test_name(self):
        adapter = OllamaAdapter()
        assert adapter.name == "ollama"

    def test_parse_param_size_billions(self):
        assert _parse_param_size("7B") == 7.0
        assert _parse_param_size("1.5B") == 1.5
        assert _parse_param_size("70B") == 70.0

    def test_parse_param_size_millions(self):
        result = _parse_param_size("355M")
        assert result is not None
        assert abs(result - 0.355) < 0.001

    def test_parse_param_size_invalid(self):
        assert _parse_param_size("unknown") is None

    def test_search_with_mocked_api(self):
        adapter = OllamaAdapter()
        mock_response = {
            "models": [
                {
                    "name": "llama3:latest",
                    "details": {"family": "llama", "parameter_size": "8B"},
                    "modified_at": "2024-01-01T00:00:00Z",
                },
                {
                    "name": "codestral:latest",
                    "details": {"family": "mistral", "parameter_size": "22B"},
                },
            ]
        }

        with patch.object(adapter, "_request", return_value=mock_response):
            results = adapter.search("llama")

        assert len(results) == 1  # only "llama" matches
        assert results[0].model_id == "llama3:latest"
        assert results[0].source == "ollama"

    def test_search_empty_query_returns_all(self):
        adapter = OllamaAdapter()
        mock_response = {
            "models": [
                {"name": "model1", "details": {}},
                {"name": "model2", "details": {}},
            ]
        }

        with patch.object(adapter, "_request", return_value=mock_response):
            results = adapter.search("")

        assert len(results) == 2

    def test_search_connection_error(self):
        adapter = OllamaAdapter()
        import urllib.error

        with patch.object(
            adapter,
            "_request",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            results = adapter.search("test")
        assert results == []

    def test_get_detail(self):
        adapter = OllamaAdapter()
        mock_response = {
            "details": {
                "family": "llama",
                "parameter_size": "8B",
                "quantization_level": "Q4_0",
            }
        }

        with patch.object(adapter, "_request", return_value=mock_response):
            inp = adapter.get_detail("llama3:latest")

        assert inp.model_id == "llama3:latest"
        assert "llama" in inp.tags

    def test_anchors_for_model(self):
        adapter = OllamaAdapter()
        details = {
            "family": "llama",
            "parameter_size": "7B",
        }
        anchors = adapter._get_anchors_for_model(details)
        assert "GGUF-available" in anchors
        assert "llama-cpp-compatible" in anchors
        assert "CPU-inference" in anchors
        assert "Llama-family" in anchors
        assert "7B-class" in anchors


class TestIntegration:
    def test_hf_adapter_to_extraction(self):
        """Test that HF adapter output feeds through extraction pipeline."""
        from model_atlas.extraction.deterministic import ModelInput
        from model_atlas.extraction.deterministic import extract as extract_det
        from model_atlas.extraction.patterns import extract as extract_pat

        inp = ModelInput(
            model_id="test/Llama-3.1-8B-Instruct",
            author="test",
            pipeline_tag="text-generation",
            tags=["text-generation", "chat", "en"],
            library_name="transformers",
            likes=100,
            downloads=50000,
            config={
                "architectures": ["LlamaForCausalLM"],
                "max_position_embeddings": 131072,
                "vocab_size": 128256,
            },
        )

        det = extract_det(inp)
        pat = extract_pat(
            model_id=inp.model_id,
            author=inp.author,
            tags=inp.tags,
            library_name=inp.library_name,
            pipeline_tag=inp.pipeline_tag,
        )

        # Verify deterministic extraction
        assert det.architecture.nodes == ["decoder-only"]
        assert det.metadata["context_length"] == ("131072", "int")
        assert det.metadata["vocab_size"] == ("128256", "int")

        det_anchors = [a[0] for a in det.anchors]
        assert "long-context-128k" in det_anchors

        # Verify pattern extraction
        pat_anchors = [a[0] for a in pat.anchors]
        assert "chat" in pat_anchors
        assert "transformers-compatible" in pat_anchors

        # Verify language tags
        assert "supported_languages" in pat.metadata
        langs = json.loads(pat.metadata["supported_languages"][0])
        assert "en" in langs
