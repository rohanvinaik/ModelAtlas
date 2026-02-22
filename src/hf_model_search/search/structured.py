"""Layer 1: Structured search via HuggingFace Hub API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from huggingface_hub import HfApi, ModelInfo

from ..config import DEFAULT_CANDIDATE_LIMIT

logger = logging.getLogger(__name__)

api = HfApi()


@dataclass
class StructuredResult:
    """A model result from the structured HF API search."""
    model_id: str
    author: str
    likes: int
    downloads: int
    pipeline_tag: str
    tags: list[str]
    library_name: str
    license: str
    card_text: str
    rank: int  # Position in API results (0-indexed)
    raw: dict = field(default_factory=dict, repr=False)


def _extract_card_text(model: ModelInfo) -> str:
    """Extract readable text from model card data."""
    parts = []
    if model.card_data:
        cd = model.card_data
        if hasattr(cd, "language") and cd.language:
            parts.append(f"Languages: {cd.language}")
        if hasattr(cd, "datasets") and cd.datasets:
            parts.append(f"Datasets: {cd.datasets}")
        if hasattr(cd, "tags") and cd.tags:
            parts.append(f"Card tags: {cd.tags}")
        # model_description is sometimes available
        if hasattr(cd, "model_description") and cd.model_description:
            parts.append(cd.model_description)
    return "\n".join(str(p) for p in parts)


def _model_to_result(model: ModelInfo, rank: int) -> StructuredResult:
    return StructuredResult(
        model_id=model.id or "",
        author=model.author or "",
        likes=model.likes or 0,
        downloads=model.downloads or 0,
        pipeline_tag=model.pipeline_tag or "",
        tags=list(model.tags or []),
        library_name=model.library_name or "",
        license=getattr(model, "license", "") or "",
        card_text=_extract_card_text(model),
        rank=rank,
        raw={
            "id": model.id,
            "sha": model.sha,
            "created_at": str(model.created_at) if model.created_at else None,
            "last_modified": str(model.last_modified) if model.last_modified else None,
        },
    )


def search(
    task: str | None = None,
    search_query: str | None = None,
    author: str | None = None,
    library: str | None = None,
    sort: str = "likes",
    limit: int = DEFAULT_CANDIDATE_LIMIT,
    min_likes: int = 0,
    min_downloads: int = 0,
) -> list[StructuredResult]:
    """
    Fetch models from the HuggingFace Hub API with structured filters.

    This is Layer 1 — hard filters that narrow the candidate pool.
    """
    # Build filter tags list: task type + library if provided
    filters: list[str] = []
    if task:
        filters.append(task)
    if library:
        filters.append(library)

    try:
        models_iter = api.list_models(
            filter=filters or None,
            search=search_query,
            author=author,
            pipeline_tag=task,
            sort=sort,
            limit=limit,
            full=True,
            cardData=True,
        )
        models = list(models_iter)
    except Exception as e:
        logger.error("HuggingFace API error: %s", e)
        return []

    results = []
    for rank, model in enumerate(models):
        if (model.likes or 0) < min_likes:
            continue
        if (model.downloads or 0) < min_downloads:
            continue
        results.append(_model_to_result(model, rank))

    logger.info("Layer 1 returned %d candidates (from %d fetched)", len(results), len(models))
    return results
