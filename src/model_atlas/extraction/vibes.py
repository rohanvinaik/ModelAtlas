"""Tier 3: Vibe summary extraction via Outlines structured generation.

Uses Outlines with a Pydantic schema to guarantee valid structured output
from a small local LLM (qwen2.5-0.5B). The model literally cannot produce
invalid output — the schema IS the parser.

For batch ingestion (ingest.py Phase C), call load_vibe_model() once at
startup, then extract_vibe_structured() per model. The model stays resident.

The old extract_vibe_summary() stub is kept for backward compatibility with
the extraction pipeline (Tier 1+2 indexing without an LLM).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy imports — these are heavy and only needed for Phase C
_outlines = None
_transformers = None


@dataclass
class VibeOutput:
    """Structured output from vibe extraction.

    Using a dataclass instead of Pydantic BaseModel so we don't add pydantic
    as a core dependency. The Outlines integration converts this to a JSON
    schema for constrained generation.
    """

    summary: str  # One sentence: what makes this model distinctive
    extra_anchors: list[str]  # 1-5 keyword tags not captured by Tier 1+2


# JSON schema for Outlines constrained generation (matches VibeOutput)
VIBE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "One sentence: what makes this model distinctive",
        },
        "extra_anchors": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "description": "1-5 keyword tags not captured by Tier 1+2",
        },
    },
    "required": ["summary", "extra_anchors"],
}

_PROMPT_TEMPLATE = """Given this ML model metadata, write a one-sentence description of what makes it distinctive, and list any additional capability tags not already captured.

Model: {model_id}
Author: {author}
Task: {pipeline_tag}
Tags: {tags}
Size: {param_count}
Family: {family}
Known capabilities: {capabilities}
Training method: {training_method}

Respond with JSON: {{"summary": "one sentence", "extra_anchors": ["tag1", "tag2"]}}"""


def build_vibe_prompt(
    model_id: str,
    author: str = "",
    pipeline_tag: str = "",
    tags: list[str] | None = None,
    param_count: str = "unknown",
    family: str = "unknown",
    capabilities: list[str] | None = None,
    training_method: str = "unknown",
) -> str:
    """Build a structured prompt from pre-extracted Tier 1+2 data."""
    tag_str = ", ".join((tags or [])[:15]) or "none"
    cap_str = ", ".join(capabilities or []) or "none"
    return _PROMPT_TEMPLATE.format(
        model_id=model_id,
        author=author,
        pipeline_tag=pipeline_tag or "unknown",
        tags=tag_str,
        param_count=param_count,
        family=family,
        capabilities=cap_str,
        training_method=training_method,
    )


class VibeExtractor:
    """Manages the Outlines model for structured vibe extraction.

    Load once at daemon startup, call extract() per model.
    """

    def __init__(self, model_name: str | None = None):
        from ..config import VIBE_MODEL_NAME

        self.model_name = model_name or VIBE_MODEL_NAME
        self._generator = None

    def load(self) -> None:  # pragma: no cover — loads ML model
        """Load the model and create the Outlines JSON generator."""
        import json

        import outlines
        import outlines.models
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading vibe model: %s", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        outlines_model = outlines.from_transformers(model, tokenizer)
        self._generator = outlines.generate.json(
            outlines_model, json.dumps(VIBE_JSON_SCHEMA)
        )
        logger.info("Vibe model loaded and ready")

    @property
    def is_loaded(self) -> bool:
        return self._generator is not None

    def extract(self, prompt: str) -> VibeOutput:
        """Run structured generation on a prompt. Returns a VibeOutput."""
        if not self._generator:
            raise RuntimeError("Model not loaded — call load() first")

        result = self._generator(prompt)

        # Outlines returns a dict matching our JSON schema
        if isinstance(result, dict):
            return VibeOutput(
                summary=result.get("summary", ""),
                extra_anchors=result.get("extra_anchors", [])[:5],
            )
        # Fallback: if Outlines returns the object directly
        return VibeOutput(
            summary=str(getattr(result, "summary", "")),
            extra_anchors=list(getattr(result, "extra_anchors", []))[:5],
        )


def extract_vibe_summary(
    model_id: str,
    card_text: str = "",
    pipeline_tag: str = "",
    tags: list[str] | None = None,
    author: str = "",
    param_count: str = "unknown",
    family: str = "unknown",
    capabilities: list[str] | None = None,
    training_method: str = "unknown",
) -> str:
    """Return empty string — vibes are delegated to Phase C / set_model_vibe.

    During Tier 1+2 batch indexing, models are indexed without vibes.
    Phase C of the ingest daemon fills them in via VibeExtractor.

    The param_count, family, and capabilities params are accepted so that
    extract_and_store can pass pre-extracted Tier 1+2 data, which
    VibeExtractor.extract() will use when Phase C is active.
    """
    return ""
