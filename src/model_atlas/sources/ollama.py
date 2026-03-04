"""Ollama local model source adapter.

Uses stdlib urllib only — no additional dependencies required.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

from ..extraction.deterministic import ModelInput
from .base import SourceAdapter, SourceSearchResult

logger = logging.getLogger(__name__)

_OLLAMA_BASE = "http://localhost:11434"
_TIMEOUT = 5  # seconds


def _parse_param_size(size_str: str) -> float | None:
    """Parse Ollama parameter size string (e.g. '7B', '1.5B') to billions."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*([BMK])", size_str, re.IGNORECASE)  # NOSONAR — linear regex
    if not match:
        return None
    val = float(match.group(1))
    unit = match.group(2).upper()
    if unit == "K":
        return val / 1_000_000
    if unit == "M":
        return val / 1_000
    return val  # B


class OllamaAdapter(SourceAdapter):
    """Source adapter for locally-running Ollama instance."""

    @property
    def name(self) -> str:
        return "ollama"

    def _request(self, method: str, path: str, data: dict | None = None) -> dict:
        """Make an HTTP request to the Ollama API."""
        url = f"{_OLLAMA_BASE}{path}"
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body else {},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: dict | None = None,
    ) -> list[SourceSearchResult]:
        """List local Ollama models, filtered by query substring."""
        try:
            resp = self._request("GET", "/api/tags")
        except (urllib.error.URLError, OSError) as e:
            logger.debug("Ollama not available: %s", e)
            return []

        models = resp.get("models", [])
        query_lower = query.lower()

        results = []
        for m in models:
            name = m.get("name", "")
            if query_lower and query_lower not in name.lower():
                continue
            details = m.get("details", {})
            results.append(
                SourceSearchResult(
                    model_id=name,
                    author="",
                    source="ollama",
                    display_name=name,
                    description=f"Family: {details.get('family', 'unknown')}",
                    downloads=0,
                    likes=0,
                    tags=[],
                    last_modified=m.get("modified_at"),
                    raw=m,
                )
            )
            if len(results) >= limit:
                break
        return results

    def get_detail(self, model_id: str) -> ModelInput:
        """Get model details from Ollama's /api/show endpoint."""
        try:
            resp = self._request("POST", "/api/show", {"name": model_id})
        except (urllib.error.URLError, OSError) as e:
            logger.warning("Ollama get_detail failed for %s: %s", model_id, e)
            return ModelInput(model_id=model_id)

        details = resp.get("details", {})

        # Build tags from Ollama metadata
        tags: list[str] = []
        family = details.get("family", "")
        if family:
            tags.append(family.lower())

        param_size = details.get("parameter_size", "")
        quant_level = details.get("quantization_level", "")

        if quant_level:
            tags.append(quant_level.lower())
            tags.append("quantized")

        # Build a synthetic config-like dict for extraction
        config: dict = {}
        if family:
            config["model_type"] = family

        # Wire parameter count from Ollama's parameter_size field
        if param_size:
            param_b = _parse_param_size(param_size)
            if param_b is not None:
                config["_parameter_count_b"] = param_b

        # Derive anchors from model details
        anchors = self._get_anchors_for_model(details)

        # Construct model input
        inp = ModelInput(
            model_id=model_id,
            author="",
            tags=tags + anchors,
            config=config if config else None,
        )

        return inp

    def _get_anchors_for_model(self, details: dict) -> list[str]:
        """Derive anchors from Ollama model details."""
        anchors = ["GGUF-available", "llama-cpp-compatible", "CPU-inference"]

        family = details.get("family", "").lower()
        family_map = {
            "llama": "Llama-family",
            "mistral": "Mistral-family",
            "gemma": "Gemma-family",
            "phi": "Phi-family",
            "qwen": "Qwen-family",
            "command-r": "Command-family",
        }
        for prefix, anchor in family_map.items():
            if prefix in family:
                anchors.append(anchor)
                break

        param_size = details.get("parameter_size", "")
        param_b = _parse_param_size(param_size)
        if param_b is not None:
            if param_b < 1:
                anchors.append("sub-1B")
                anchors.append("edge-deployable")
            elif param_b < 4:
                anchors.append("3B-class")
                anchors.append("consumer-GPU-viable")
            elif param_b < 10:
                anchors.append("7B-class")
            elif param_b < 20:
                anchors.append("13B-class")

        return anchors
