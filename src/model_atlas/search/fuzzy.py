"""Layer 2: Fuzzy string matching on model metadata."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Common stop words to skip during token matching
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "for",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "no",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "also",
        "at",
        "before",
        "between",
        "by",
        "down",
        "during",
        "from",
        "in",
        "into",
        "of",
        "on",
        "out",
        "over",
        "then",
        "there",
        "through",
        "to",
        "under",
        "up",
        "with",
        "that",
        "this",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "it",
        "its",
        "model",
        "models",  # too generic in this context
    }
)


@dataclass
class FuzzyScore:
    model_id: str
    score: float  # 0.0 to 1.0
    best_match_field: str  # Which field had the best match
    best_match_value: str  # The matched value


def _tokenize_query(query: str) -> list[str]:
    """Extract meaningful tokens from a natural language query."""
    tokens = re.findall(r"[a-zA-Z0-9_.-]+", query.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def _build_searchable_strings(
    model_id: str, tags: list[str], pipeline_tag: str, card_text: str
) -> dict[str, str]:
    """Build a dict of field_name -> searchable string for a model."""
    # Split model_id into parts: "meta-llama/Llama-3.1-8B-Instruct" -> "meta llama Llama 3 1 8B Instruct"
    id_parts = re.findall(r"[a-zA-Z0-9]+", model_id)
    return {
        "model_id": " ".join(id_parts).lower(),
        "tags": " ".join(tags).lower(),
        "pipeline_tag": pipeline_tag.lower(),
        "card_text": card_text[:500].lower(),
    }


def score_models(
    query: str,
    models: list[dict],
) -> list[FuzzyScore]:
    """
    Score models against a query using fuzzy string matching.

    Each model dict must have: model_id, tags, pipeline_tag, card_text

    Strategy:
    1. Tokenize the query into meaningful terms
    2. For each model, build searchable strings from its fields
    3. Score each query token against each field using token_set_ratio
    4. The model's score is the average of its best per-token scores
    """
    tokens = _tokenize_query(query)
    if not tokens:
        return [FuzzyScore(m["model_id"], 0.0, "", "") for m in models]

    results = []
    for model in models:
        fields = _build_searchable_strings(
            model["model_id"],
            model.get("tags", []),
            model.get("pipeline_tag", ""),
            model.get("card_text", ""),
        )

        # For each query token, find the best matching field
        token_scores = []
        best_field = ""
        best_value = ""
        best_score = 0.0

        for token in tokens:
            token_best = 0.0
            token_field = ""
            token_value = ""
            for field_name, field_text in fields.items():
                if not field_text:
                    continue
                # Use token_set_ratio for partial/reordered matching
                s = fuzz.token_set_ratio(token, field_text) / 100.0
                # Bonus for exact substring match
                if token in field_text:
                    s = min(1.0, s + 0.2)
                if s > token_best:
                    token_best = s
                    token_field = field_name
                    token_value = field_text[:80]
            token_scores.append(token_best)
            if token_best > best_score:
                best_score = token_best
                best_field = token_field
                best_value = token_value

        avg_score = sum(token_scores) / len(token_scores) if token_scores else 0.0

        results.append(
            FuzzyScore(
                model_id=model["model_id"],
                score=avg_score,
                best_match_field=best_field,
                best_match_value=best_value,
            )
        )

    return results
