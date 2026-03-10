"""SQLite database for the semantic network.

Stores models positioned across 8 orthogonal semantic banks, connected
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

# The eight semantic banks
BANKS = (
    "ARCHITECTURE",
    "CAPABILITY",
    "EFFICIENCY",
    "COMPATIBILITY",
    "LINEAGE",
    "DOMAIN",
    "QUALITY",
    "TRAINING",
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
    "TRAINING": "standard supervised fine-tuning",
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

_PHASE_D_SCHEMA = """
CREATE TABLE IF NOT EXISTS phase_d_runs (
    run_id      TEXT PRIMARY KEY,
    phase       TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    config      TEXT,
    status      TEXT DEFAULT 'running',
    summary     TEXT
);

CREATE TABLE IF NOT EXISTS audit_findings (
    finding_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT REFERENCES phase_d_runs(run_id),
    model_id      TEXT REFERENCES models(model_id),
    mismatch_type TEXT NOT NULL,
    bank          TEXT,
    c2_anchor     TEXT,
    det_anchor    TEXT,
    severity      REAL DEFAULT 0.5,
    detail        TEXT
);

CREATE TABLE IF NOT EXISTS correction_events (
    event_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT REFERENCES phase_d_runs(run_id),
    model_id          TEXT REFERENCES models(model_id),
    tier              TEXT NOT NULL,
    original_prompt   TEXT,
    original_response TEXT,
    healed_response   TEXT,
    anchors_added     TEXT,
    anchors_removed   TEXT,
    rationale         TEXT,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_model ON audit_findings(model_id);
CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_findings(run_id);
CREATE INDEX IF NOT EXISTS idx_correction_model ON correction_events(model_id);
CREATE INDEX IF NOT EXISTS idx_correction_run ON correction_events(run_id);
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
        if own_conn:  # pragma: no cover — only when no conn passed
            c.close()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after initial schema (safe to re-run)."""
    # anchors.source (added for anchor provenance)
    try:
        conn.execute("SELECT source FROM anchors LIMIT 1")
    except sqlite3.OperationalError:  # pragma: no cover — legacy schema only
        conn.execute("ALTER TABLE anchors ADD COLUMN source TEXT DEFAULT 'bootstrap'")

    # model_anchors.confidence (added for confidence-weighted scoring)
    try:
        conn.execute("SELECT confidence FROM model_anchors LIMIT 1")
    except sqlite3.OperationalError:  # pragma: no cover — legacy schema only
        conn.execute("ALTER TABLE model_anchors ADD COLUMN confidence REAL DEFAULT 1.0")


def init_db(conn: sqlite3.Connection | None = None) -> None:
    """Create tables and bootstrap anchor dictionary."""
    with transaction(conn) as c:
        c.executescript(_SCHEMA)
        c.executescript(_PHASE_D_SCHEMA)
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


# --- Phase D Helpers ---


def create_phase_d_run(
    conn: sqlite3.Connection,
    phase: str,
    config: dict | None = None,
) -> str:
    """Create a new Phase D run record. Returns run_id (UUID)."""
    import uuid
    from datetime import datetime, timezone

    run_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO phase_d_runs (run_id, phase, started_at, config, status)
           VALUES (?, ?, ?, ?, 'running')""",
        (
            run_id,
            phase,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(config) if config else None,
        ),
    )
    return run_id


def finish_phase_d_run(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
    summary: dict | None = None,
) -> None:
    """Update a Phase D run with completion status."""
    from datetime import datetime, timezone

    conn.execute(
        """UPDATE phase_d_runs
           SET finished_at = ?, status = ?, summary = ?
           WHERE run_id = ?""",
        (
            datetime.now(timezone.utc).isoformat(),
            status,
            json.dumps(summary) if summary else None,
            run_id,
        ),
    )


def insert_audit_finding(
    conn: sqlite3.Connection,
    run_id: str,
    model_id: str,
    mismatch_type: str,
    bank: str | None = None,
    c2_anchor: str | None = None,
    det_anchor: str | None = None,
    severity: float = 0.5,
    detail: dict | None = None,
) -> int:
    """Insert an audit finding. Returns finding_id."""
    cursor = conn.execute(
        """INSERT INTO audit_findings
           (run_id, model_id, mismatch_type, bank, c2_anchor, det_anchor, severity, detail)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            model_id,
            mismatch_type,
            bank,
            c2_anchor,
            det_anchor,
            severity,
            json.dumps(detail) if detail else None,
        ),
    )
    return cursor.lastrowid or 0


def insert_correction_event(
    conn: sqlite3.Connection,
    run_id: str,
    model_id: str,
    tier: str,
    original_prompt: str | None = None,
    original_response: str | None = None,
    healed_response: str | None = None,
    anchors_added: list[str] | None = None,
    anchors_removed: list[str] | None = None,
    rationale: str | None = None,
) -> int:
    """Insert a correction event. Returns event_id."""
    from datetime import datetime, timezone

    cursor = conn.execute(
        """INSERT INTO correction_events
           (run_id, model_id, tier, original_prompt, original_response,
            healed_response, anchors_added, anchors_removed, rationale, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            model_id,
            tier,
            original_prompt,
            original_response,
            healed_response,
            json.dumps(anchors_added) if anchors_added else None,
            json.dumps(anchors_removed) if anchors_removed else None,
            rationale,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    return cursor.lastrowid or 0


# --- Re-exports for backward compatibility ---
from .db_ingest import (  # noqa: E402
    get_connection as get_ingest_connection,
)
from .db_ingest import (  # noqa: E402
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
    "create_phase_d_run",
    "finish_phase_d_run",
    "insert_audit_finding",
    "insert_correction_event",
]
