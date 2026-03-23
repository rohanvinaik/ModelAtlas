"""Tests for ingest_phase_c."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src.model_atlas.ingest_phase_c import (
    export_c1,
    export_c2,
    export_c3,
    get_phase_c_status,
    print_phase_c_status,
    select_summaries,
)
