"""Tests for server."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from src.model_atlas.server import hf_build_index, hf_compare_models, hf_get_model_detail, hf_index_status, hf_search_models, list_model_sources, main, navigate_models, search_models, set_model_vibe


