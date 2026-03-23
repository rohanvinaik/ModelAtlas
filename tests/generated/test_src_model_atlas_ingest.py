"""Tests for ingest."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from src.model_atlas.ingest import get_status, phase_a, phase_b, print_status, run
