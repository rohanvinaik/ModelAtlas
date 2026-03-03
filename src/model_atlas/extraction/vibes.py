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
    selected_anchors: list[str]  # 0-5 anchors selected from the dictionary


# JSON schema for Outlines constrained generation (matches VibeOutput)
VIBE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "minLength": 20,
            "description": "One sentence: what makes this model distinctive",
        },
        "selected_anchors": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 3,
                "pattern": "^[a-z][a-z0-9-]+$",
            },
            "minItems": 0,
            "maxItems": 5,
            "description": "0-5 anchors selected from the provided dictionary lists",
        },
    },
    "required": ["summary", "selected_anchors"],
}

_PROMPT_TEMPLATE = """You are an ML model classifier. Given metadata about a model:
1. Write a one-sentence summary of what makes this model distinctive.
2. Select which additional anchors from the lists below apply to this model.

Model: {model_id}
Author: {author}
Task: {pipeline_tag}
Tags: {tags}
Size: {param_count}
Family: {family}
Already assigned: {existing_anchors}
Config: {config_summary}
Card excerpt: {card_excerpt}

Select from these CAPABILITY anchors (only ones NOT already assigned):
{capability_candidates}

Select from these DOMAIN anchors (only ones NOT already assigned):
{domain_candidates}

Respond with valid JSON:
- "summary": one sentence
- "selected_anchors": array of anchor labels from the lists above (empty if none apply)"""


def build_vibe_prompt(
    model_id: str,
    author: str = "",
    pipeline_tag: str = "",
    tags: list[str] | None = None,
    param_count: str = "unknown",
    family: str = "unknown",
    existing_anchors: list[str] | None = None,
    config_summary: str = "",
    card_excerpt: str = "",
    capability_candidates: list[str] | None = None,
    domain_candidates: list[str] | None = None,
) -> str:
    """Build a selection prompt from pre-extracted Tier 1+2 data.

    The model selects from curated CAPABILITY and DOMAIN anchor lists
    rather than generating free-form tags.
    """
    tag_str = ", ".join((tags or [])[:15]) or "none"
    anchor_str = ", ".join(existing_anchors or []) or "none"
    cap_cands = ", ".join(capability_candidates or []) or "none"
    dom_cands = ", ".join(domain_candidates or []) or "none"
    return _PROMPT_TEMPLATE.format(
        model_id=model_id,
        author=author,
        pipeline_tag=pipeline_tag or "unknown",
        tags=tag_str,
        param_count=param_count,
        family=family,
        existing_anchors=anchor_str,
        config_summary=config_summary or "none",
        card_excerpt=card_excerpt or "none",
        capability_candidates=cap_cands,
        domain_candidates=dom_cands,
    )


_QUALITY_GATE_TEMPLATE = """You are a blind quality reviewer for ML model summaries. Given ONLY a model ID, its summary, and its tags, score the summary on three axes (0-3 each):

- specificity: Does the summary mention concrete distinguishing details (architecture, dataset, size, technique)? 0=generic/boilerplate, 3=highly specific and informative.
- coherence: Is the summary well-formed, grammatical, and internally consistent? 0=garbled/contradictory, 3=clear and professional.
- artifacts: Does the summary contain LLM artifacts (repetition, hallucinated URLs, prompt leakage, filler phrases)? 0=severe artifacts, 3=clean.

Also list any flags (empty list if none): "generic" (could apply to any model), "hallucinated" (claims not supported by tags), "truncated" (sentence cut off), "repetitive" (repeated phrases).

Model: {model_id}
Summary: {summary}
Tags: {tags}

Respond with valid JSON containing keys: "specificity" (int 0-3), "coherence" (int 0-3), "artifacts" (int 0-3), "flags" (array of strings, empty if none)."""


def build_quality_gate_prompt(
    model_id: str,
    summary: str,
    tags: list[str] | None = None,
) -> str:
    """Build a blind quality review prompt for Phase C3.

    Only model_id, summary, and tags are provided — no source material.
    This forces the reviewer to evaluate the summary on its own merits.
    """
    tag_str = ", ".join((tags or [])[:15]) or "none"
    return _QUALITY_GATE_TEMPLATE.format(
        model_id=model_id,
        summary=summary,
        tags=tag_str,
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
                selected_anchors=result.get("selected_anchors", [])[:5],
            )
        # Fallback: if Outlines returns the object directly
        return VibeOutput(
            summary=str(getattr(result, "summary", "")),
            selected_anchors=list(getattr(result, "selected_anchors", []))[:5],
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
