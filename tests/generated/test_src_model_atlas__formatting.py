"""Tests for _formatting."""

from __future__ import annotations

import pytest
from src.model_atlas._formatting import (
    candidates_to_dicts,
    fetch_from_hf_api,
    format_fuzzy_results,
    format_network_results,
    structured_to_dict,
)
