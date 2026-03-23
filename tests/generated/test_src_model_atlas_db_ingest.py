"""Tests for db_ingest."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from src.model_atlas.db_ingest import get_connection, init_db


