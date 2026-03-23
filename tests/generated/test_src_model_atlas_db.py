"""Tests for db."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src.model_atlas.db import (
    add_link,
    create_phase_d_run,
    finish_phase_d_run,
    get_connection,
    get_or_create_anchor,
    init_db,
    insert_audit_finding,
    insert_correction_event,
    insert_model,
    link_anchor,
    set_metadata,
    set_position,
    transaction,
)
