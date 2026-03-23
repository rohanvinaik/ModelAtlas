"""Tests for db_ingest."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src.model_atlas.db_ingest import get_connection, init_db
