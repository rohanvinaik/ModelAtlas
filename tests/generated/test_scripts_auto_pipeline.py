"""Tests for auto_pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from scripts.auto_pipeline import (
    check_ollama,
    deploy_worker,
    ensure_workdirs,
    log,
    main,
    poll_workers,
    run_local_python,
    scp_from,
    scp_to,
    ssh,
    stage_c2,
    stage_c3,
    stage_d1,
    stage_d2,
    stage_d3,
    stage_d4,
    stage_summary_selection,
    start_worker,
)
