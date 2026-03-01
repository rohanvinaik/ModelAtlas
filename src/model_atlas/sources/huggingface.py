"""HuggingFace Hub source adapter."""

from __future__ import annotations

import logging
import tempfile

from huggingface_hub import HfApi, hf_hub_download

from ..extraction.deterministic import ModelInput
from .base import SourceAdapter, SourceSearchResult

logger = logging.getLogger(__name__)


class HuggingFaceAdapter(SourceAdapter):
    """Source adapter for the HuggingFace Hub."""

    def __init__(self) -> None:
        self._api = HfApi()

    @property
    def name(self) -> str:
        return "huggingface"

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: dict | None = None,
    ) -> list[SourceSearchResult]:
        """Search HuggingFace Hub models."""
        filters = filters or {}
        try:
            models = list(
                self._api.list_models(
                    search=query,
                    author=filters.get("author"),
                    pipeline_tag=filters.get("task"),
                    sort="likes",
                    limit=limit,
                    full=True,
                )
            )
        except Exception as e:
            logger.error("HuggingFace search error: %s", e)
            return []

        results = []
        for model in models:
            results.append(
                SourceSearchResult(
                    model_id=model.id or "",
                    author=model.author or "",
                    source="huggingface",
                    display_name=(model.id or "").split("/")[-1],
                    description="",
                    downloads=model.downloads or 0,
                    likes=model.likes or 0,
                    tags=list(model.tags or []),
                    last_modified=(
                        str(model.last_modified) if model.last_modified else None
                    ),
                    raw={"id": model.id, "sha": model.sha},
                )
            )
        return results

    def get_detail(self, model_id: str) -> ModelInput:
        """Get full model details from HuggingFace."""
        info = self._api.model_info(model_id, cardData=True)
        config = self.fetch_config(model_id)
        return ModelInput(
            model_id=info.id or model_id,
            author=info.author or "",
            pipeline_tag=info.pipeline_tag or "",
            tags=list(info.tags or []),
            library_name=info.library_name or "",
            likes=info.likes or 0,
            downloads=info.downloads or 0,
            created_at=str(info.created_at) if info.created_at else None,
            license_str=getattr(info, "license", "") or "",
            safetensors_info=(
                info.safetensors if hasattr(info, "safetensors") else None
            ),
            config=config,
        )

    def fetch_config(self, model_id: str) -> dict | None:
        """Download and parse config.json from HuggingFace."""
        try:
            import json

            path = hf_hub_download(
                model_id,
                "config.json",
                cache_dir=tempfile.gettempdir(),
            )
            with open(path) as f:
                return json.load(f)
        except Exception:
            logger.debug("No config.json for %s", model_id)
            return None
