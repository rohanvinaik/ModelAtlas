"""Tests for pipeline."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from src.model_atlas.extraction.pipeline import extract_and_store, extract_batch, infer_relationships


