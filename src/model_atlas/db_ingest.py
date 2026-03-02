"""Ingest progress tracking database.

Separate from the main semantic network DB — tracks which models have
been fetched, extracted, and vibed during the ingest pipeline.
"""

from __future__ import annotations

import sqlite3

from .config import INGEST_DB_PATH
from .db import _ensure_db_dir

_INGEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_models (
    model_id         TEXT PRIMARY KEY,
    source           TEXT DEFAULT 'huggingface',
    likes            INTEGER DEFAULT 0,
    phase_a_done     INTEGER DEFAULT 0,
    phase_b_done     INTEGER DEFAULT 0,
    phase_c_done     INTEGER DEFAULT 0,
    phase_c_attempts INTEGER DEFAULT 0,
    raw_json         TEXT,
    fetched_at       TEXT,
    extracted_at     TEXT,
    vibed_at         TEXT
);
"""


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Get a connection to the ingest state database."""
    path = db_path or str(INGEST_DB_PATH)
    _ensure_db_dir()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create ingest progress tracking tables."""
    conn.executescript(_INGEST_SCHEMA)
    conn.commit()
