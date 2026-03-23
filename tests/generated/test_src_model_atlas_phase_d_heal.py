"""Tests for phase_d_heal."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from src.model_atlas.phase_d_heal import HealExportResult, build_healing_prompt, export_d3, select_healing_candidates
