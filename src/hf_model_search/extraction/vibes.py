"""Tier 3: Heuristic NLP extraction from model card text.

Generates the vibe_summary — one sentence capturing the irreducible
"feel" of a model. This is the ONLY prose stored in the network.
"""

from __future__ import annotations

import re


def _extract_first_sentence(text: str) -> str:
    """Pull the first meaningful sentence from model card text."""
    # Skip markdown headers and badges
    lines = text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("#", "!", "[", "<", "---", "```", "|")):
            continue
        if len(line) < 20:
            continue
        # Found a content line — extract first sentence
        match = re.match(r"(.+?[.!?])\s", line)
        if match:
            return match.group(1).strip()
        return line[:200].strip()
    return ""


def _build_vibe_from_signals(
    model_id: str,
    pipeline_tag: str,
    tags: list[str],
    author: str,
) -> str:
    """Synthesize a vibe summary from structured signals when card text is sparse."""
    parts: list[str] = []

    name = model_id.split("/")[-1] if "/" in model_id else model_id
    parts.append(name)

    if author:
        parts.append(f"by {author}")

    descriptors: list[str] = []
    name_lower = name.lower()
    tag_str = " ".join(tags).lower()

    if "instruct" in name_lower:
        descriptors.append("instruction-tuned")
    elif "chat" in name_lower:
        descriptors.append("chat-optimized")
    if "code" in name_lower or "coder" in name_lower:
        descriptors.append("code-focused")
    if "vision" in tag_str or "vlm" in name_lower:
        descriptors.append("vision-capable")

    if pipeline_tag:
        descriptors.append(pipeline_tag.replace("-", " "))

    if descriptors:
        parts.append("— " + ", ".join(descriptors))

    return " ".join(parts)


def extract_vibe_summary(
    model_id: str,
    card_text: str = "",
    pipeline_tag: str = "",
    tags: list[str] | None = None,
    author: str = "",
) -> str:
    """Generate a one-sentence vibe summary for a model.

    Tries the model card first; falls back to structured signal synthesis.
    """
    tags = tags or []

    if card_text and len(card_text) > 50:
        sentence = _extract_first_sentence(card_text)
        if sentence and len(sentence) > 20:
            # Truncate to a reasonable length
            if len(sentence) > 200:
                sentence = sentence[:197] + "..."
            return sentence

    return _build_vibe_from_signals(model_id, pipeline_tag, tags, author)
