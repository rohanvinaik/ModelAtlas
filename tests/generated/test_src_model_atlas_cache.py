"""Tests for cache."""

from __future__ import annotations

import pytest
from src.model_atlas.cache import (
    clear_cache,
    get_cached_card_text,
    get_cached_model,
    store_card_text,
    store_model,
)
