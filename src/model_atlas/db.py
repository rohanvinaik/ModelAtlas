"""SQLite database for the semantic network.

Stores models positioned across 7 orthogonal semantic banks, connected
through a shared anchor dictionary. This is the primary storage — the
model_metadata table is overflow for data that doesn't decompose into
the network.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from .config import NETWORK_DB_PATH
from .db_bootstrap import BOOTSTRAP_ANCHORS

# The seven semantic banks
BANKS = (
    "ARCHITECTURE",
    "CAPABILITY",
    "EFFICIENCY",
    "COMPATIBILITY",
    "LINEAGE",
    "DOMAIN",
    "QUALITY",
)

# Zero states for each bank (the semantic origin)
ZERO_STATES = {
    "ARCHITECTURE": "standard transformer decoder",
    "CAPABILITY": "general language model",
    "EFFICIENCY": "~7B mainstream",
    "COMPATIBILITY": "standard transformers + PyTorch",
    "LINEAGE": "base/foundational model",
    "DOMAIN": "general knowledge",
    "QUALITY": "established, mainstream adoption",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id    TEXT PRIMARY KEY,
    author      TEXT,
    source      TEXT DEFAULT 'huggingface',
    display_name TEXT
);

CREATE TABLE IF NOT EXISTS model_positions (
    model_id    TEXT REFERENCES models(model_id),
    bank        TEXT,
    path_sign   INTEGER,
    path_depth  INTEGER,
    path_nodes  TEXT,
    zero_state  TEXT,
    PRIMARY KEY (model_id, bank)
);

CREATE TABLE IF NOT EXISTS model_links (
    source_id   TEXT REFERENCES models(model_id),
    target_id   TEXT REFERENCES models(model_id),
    relation    TEXT,
    weight      REAL DEFAULT 1.0,
    PRIMARY KEY (source_id, target_id, relation)
);

CREATE TABLE IF NOT EXISTS anchors (
    anchor_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT UNIQUE,
    bank        TEXT,
    category    TEXT,
    source      TEXT DEFAULT 'bootstrap'
);

CREATE TABLE IF NOT EXISTS model_anchors (
    model_id    TEXT REFERENCES models(model_id),
    anchor_id   INTEGER REFERENCES anchors(anchor_id),
    weight      REAL DEFAULT 1.0,
    confidence  REAL DEFAULT 1.0,
    PRIMARY KEY (model_id, anchor_id)
);

CREATE TABLE IF NOT EXISTS model_metadata (
    model_id    TEXT REFERENCES models(model_id),
    key         TEXT,
    value       TEXT,
    value_type  TEXT,
    PRIMARY KEY (model_id, key)
);

CREATE INDEX IF NOT EXISTS idx_positions_bank ON model_positions(bank);
CREATE INDEX IF NOT EXISTS idx_links_source ON model_links(source_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON model_links(target_id);
CREATE INDEX IF NOT EXISTS idx_model_anchors_model ON model_anchors(model_id);
CREATE INDEX IF NOT EXISTS idx_model_anchors_anchor ON model_anchors(anchor_id);
CREATE INDEX IF NOT EXISTS idx_metadata_key ON model_metadata(key);
"""


def _ensure_db_dir() -> None:
    NETWORK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Get a connection to the network database."""
    _ensure_db_dir()
    conn = sqlite3.connect(str(NETWORK_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager for database transactions."""
    own_conn = conn is None
    c = conn if conn is not None else get_connection()
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        if own_conn:
            c.close()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after initial schema (safe to re-run)."""
    # anchors.source (added for anchor provenance)
    try:
        conn.execute("SELECT source FROM anchors LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE anchors ADD COLUMN source TEXT DEFAULT 'bootstrap'")

    # model_anchors.confidence (added for confidence-weighted scoring)
    try:
        conn.execute("SELECT confidence FROM model_anchors LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE model_anchors ADD COLUMN confidence REAL DEFAULT 1.0")


def init_db(conn: sqlite3.Connection | None = None) -> None:
    """Create tables and bootstrap anchor dictionary."""
    with transaction(conn) as c:
        c.executescript(_SCHEMA)
        _migrate_schema(c)
        # Bootstrap anchors (skip duplicates)
        c.executemany(
            "INSERT OR IGNORE INTO anchors (label, bank, category, source) VALUES (?, ?, ?, 'bootstrap')",
            BOOTSTRAP_ANCHORS,
        )


# --- CRUD Operations ---


def insert_model(
    conn: sqlite3.Connection,
    model_id: str,
    author: str = "",
    source: str = "huggingface",
    display_name: str = "",
) -> None:
    """Insert or update a model entity."""
    conn.execute(
        """INSERT INTO models (model_id, author, source, display_name)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(model_id) DO UPDATE SET
               author=excluded.author,
               source=excluded.source,
               display_name=excluded.display_name""",
        (model_id, author, source, display_name or model_id.split("/")[-1]),
    )


def set_position(
    conn: sqlite3.Connection,
    model_id: str,
    bank: str,
    path_sign: int,
    path_depth: int,
    path_nodes: list[str] | None = None,
) -> None:
    """Set a model's position in a semantic bank."""
    conn.execute(
        """INSERT INTO model_positions (model_id, bank, path_sign, path_depth, path_nodes, zero_state)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(model_id, bank) DO UPDATE SET
               path_sign=excluded.path_sign,
               path_depth=excluded.path_depth,
               path_nodes=excluded.path_nodes""",
        (
            model_id,
            bank,
            path_sign,
            path_depth,
            json.dumps(path_nodes) if path_nodes else None,
            ZERO_STATES.get(bank, ""),
        ),
    )


def add_link(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    relation: str,
    weight: float = 1.0,
) -> None:
    """Add an explicit relationship between two models."""
    conn.execute(
        """INSERT OR REPLACE INTO model_links (source_id, target_id, relation, weight)
           VALUES (?, ?, ?, ?)""",
        (source_id, target_id, relation, weight),
    )


def get_or_create_anchor(
    conn: sqlite3.Connection,
    label: str,
    bank: str,
    category: str = "",
    source: str = "bootstrap",
) -> int:
    """Get an anchor's ID, creating it if it doesn't exist.

    If the anchor exists with a different bank, logs a warning and keeps
    the original bank assignment to avoid semantic drift.
    """
    row = conn.execute(
        "SELECT anchor_id, bank FROM anchors WHERE label = ?", (label,)
    ).fetchone()
    if row:
        if row["bank"] != bank:
            import logging

            logging.getLogger(__name__).warning(
                "Anchor '%s' already assigned to bank '%s', ignoring reassignment to '%s'",
                label,
                row["bank"],
                bank,
            )
        return row["anchor_id"]
    cursor = conn.execute(
        "INSERT INTO anchors (label, bank, category, source) VALUES (?, ?, ?, ?)",
        (label, bank, category, source),
    )
    row_id = cursor.lastrowid
    if row_id is None:  # pragma: no cover — always set after INSERT
        raise RuntimeError("INSERT did not return a lastrowid")
    return int(row_id)


def link_anchor(
    conn: sqlite3.Connection,
    model_id: str,
    anchor_id: int,
    weight: float = 1.0,
    confidence: float = 1.0,
) -> None:
    """Link a model to an anchor with optional confidence score."""
    conn.execute(
        """INSERT OR REPLACE INTO model_anchors (model_id, anchor_id, weight, confidence)
           VALUES (?, ?, ?, ?)""",
        (model_id, anchor_id, weight, confidence),
    )


def set_metadata(
    conn: sqlite3.Connection,
    model_id: str,
    key: str,
    value: str,
    value_type: str = "str",
) -> None:
    """Set overflow metadata for a model."""
    conn.execute(
        """INSERT OR REPLACE INTO model_metadata (model_id, key, value, value_type)
           VALUES (?, ?, ?, ?)""",
        (model_id, key, value, value_type),
    )


# --- Re-exports for backward compatibility ---
from .db_ingest import (  # noqa: E402
    get_connection as get_ingest_connection,
    init_db as init_ingest_db,
)
from .db_queries import (  # noqa: E402
    batch_get_anchor_sets,
    batch_get_positions,
    compute_anchor_idf,
    find_models_by_anchor,
    find_models_by_bank_range,
    get_anchor_set,
    get_model,
    network_stats,
)

__all__ = [
    "BANKS",
    "ZERO_STATES",
    "get_connection",
    "transaction",
    "init_db",
    "insert_model",
    "set_position",
    "add_link",
    "get_or_create_anchor",
    "link_anchor",
    "set_metadata",
    "get_model",
    "get_anchor_set",
    "find_models_by_anchor",
    "find_models_by_bank_range",
    "compute_anchor_idf",
    "batch_get_positions",
    "batch_get_anchor_sets",
    "network_stats",
    "get_ingest_connection",
    "init_ingest_db",
]
