"""Tier 3: Vibe summary extraction.

The LLM calling the MCP tools IS the NLP processor. This module provides
only a minimal fallback for batch indexing — the real vibes come from the
LLM via the set_model_vibe tool.
"""

from __future__ import annotations


def extract_vibe_summary(
    model_id: str,
    card_text: str = "",
    pipeline_tag: str = "",
    tags: list[str] | None = None,
    author: str = "",
) -> str:
    """Return empty string — vibes are delegated to the LLM via set_model_vibe.

    During batch indexing, models are indexed without vibes. The build_index
    response includes a vibes_needed list so the LLM can fill them in.
    """
    return ""
