"""Tests for backfill_raw_json."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from scripts.backfill_raw_json import fetch_with_retry, main, model_info_to_raw


