"""Tests for phase_d_training."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src.model_atlas.phase_d_training import (
    TrainingStats,
    export_training_data,
    get_training_data_stats,
)
