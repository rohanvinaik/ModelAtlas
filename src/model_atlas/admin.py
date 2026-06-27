"""Audit-logged single-field write primitive for canonical reference data.

Implements the discipline described in
``PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md`` §7-§8: every canonical-data
write funnels through one function with a required ``reason``, dry-run by
default, and an append-only JSONL audit log that lives outside the database
it audits.

This module is *only* the primitive. It does not re-route existing writes —
that migration is a separate effort. New code paths that mutate canonical
reference tables should use ``patch_field``; legacy paths remain unchanged
until explicitly migrated.

Canonical vs. observation tables
--------------------------------
``CANONICAL_TABLES`` lists the reference tier (Mode 1 in the doc). Other
tables (``model_metadata``, ``audit_findings``, ``correction_events``,
``phase_d_runs``) are observation/derived (Mode 2) and accept direct writes.

Audit log location
------------------
For the canonical network DB at ``~/.cache/model-atlas/network.db`` the log
goes to ``<repo>/data/patches.jsonl`` (git-tracked durability, doc §67). For
test/staging DBs at other paths the log is co-located: ``<db_dir>/patches.jsonl``.
``MODEL_ATLAS_PATCHES_PATH`` overrides both and is required for ``:memory:``
connections.
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import NETWORK_DB_PATH

# Canonical reference tables. Writes here must go through patch_field with a
# sourced reason. Adding a table is an architectural decision — coordinate
# with the bi-modal split documented in docs/admin.md.
CANONICAL_TABLES: frozenset[str] = frozenset(
    {
        "models",
        "model_positions",
        "model_links",
        "anchors",
    }
)

# Repo-root default audit log location. The audit log lives in-repo so it is
# git-tracked and survives a corrupted network.db (doc §67).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_LOG_PATH = _REPO_ROOT / "data" / "patches.jsonl"

_PATCHES_PATH_ENV = "MODEL_ATLAS_PATCHES_PATH"


class PatchError(ValueError):
    """Raised when a patch_field call violates its contract."""


@dataclass(frozen=True)
class PatchResult:
    """The outcome of a patch_field call.

    ``applied`` is True only when ``apply=True`` was passed *and* the value
    actually changed. ``unchanged`` is True when old == new (no write
    performed and no audit entry written, regardless of ``apply``).
    """

    table: str
    key: dict[str, Any]
    field: str
    old_value: Any
    new_value: Any
    reason: str
    applied: bool
    audit_log_path: Path
    unchanged: bool = False


def _connection_main_path(conn: sqlite3.Connection) -> str | None:
    """Return the file path of the connection's main database, or None for :memory:."""
    for row in conn.execute("PRAGMA database_list"):
        # PRAGMA database_list returns (seq, name, file)
        name = row[1]
        path = row[2]
        if name == "main":
            return path or None
    return None


def _audit_log_path_for(conn: sqlite3.Connection) -> Path:
    """Derive the audit log path from the connection's main DB path.

    - Canonical NETWORK_DB_PATH → in-repo data/patches.jsonl (doc §7).
    - Other file-backed DBs (tests using tmp_path, staging) → patches.jsonl
      co-located with the DB file.
    - :memory: → require MODEL_ATLAS_PATCHES_PATH override.
    """
    override = os.environ.get(_PATCHES_PATH_ENV)
    if override:
        return Path(override)

    main_path = _connection_main_path(conn)
    if not main_path:
        raise PatchError(
            "Cannot derive audit log path for in-memory connection; "
            f"set the {_PATCHES_PATH_ENV} environment variable or pass "
            "audit_log_path explicitly."
        )

    db_path = Path(main_path).resolve()
    try:
        canonical = NETWORK_DB_PATH.resolve()
    except OSError:  # pragma: no cover — only if NETWORK_DB_PATH parent is unreadable
        canonical = NETWORK_DB_PATH
    if db_path == canonical:
        return DEFAULT_AUDIT_LOG_PATH
    return db_path.parent / "patches.jsonl"


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _count_matching(
    conn: sqlite3.Connection, table: str, key: dict[str, Any]
) -> int:
    where = " AND ".join(f'"{c}" = ?' for c in key)
    sql = f'SELECT COUNT(*) FROM "{table}" WHERE {where}'
    return int(conn.execute(sql, tuple(key.values())).fetchone()[0])


def _fetch_current(
    conn: sqlite3.Connection, table: str, key: dict[str, Any], col: str
) -> Any:
    where = " AND ".join(f'"{c}" = ?' for c in key)
    sql = f'SELECT "{col}" FROM "{table}" WHERE {where}'
    row = conn.execute(sql, tuple(key.values())).fetchone()
    if row is None:
        return None
    # row may be sqlite3.Row (mapping) or tuple; column 0 is the value either way
    return row[0]


def _utc_iso_z() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def patch_field(
    table: str,
    key: dict[str, Any],
    field: str,
    value: Any,
    reason: str,
    *,
    apply: bool = False,
    conn: sqlite3.Connection,
    audit_log_path: Path | None = None,
) -> PatchResult:
    """Patch a single field on a single canonical row, with audit logging.

    Contract (see ``PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md`` §8):

    - ``table`` must be in :data:`CANONICAL_TABLES`.
    - ``key`` must uniquely identify exactly one existing row.
    - ``field`` must be a real column on ``table`` and not one of the
      ``key`` columns.
    - ``reason`` must be a non-empty string after whitespace stripping.
    - Default is dry-run: returns the diff without modifying anything.
    - When ``apply=True`` and the value actually changes, the UPDATE runs
      and a JSONL line is appended to the audit log. Both happen inside
      the caller's transaction (if any); the caller commits.
    - When old == new, no UPDATE and no audit entry are made; the result
      carries ``unchanged=True``.

    The function never commits on the caller's behalf. Callers that want
    durability should wrap calls in ``with db.transaction(conn): ...``.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise PatchError("reason must be a non-empty string")

    if table not in CANONICAL_TABLES:
        raise PatchError(
            f"table {table!r} is not canonical; "
            f"allowed: {sorted(CANONICAL_TABLES)}"
        )

    if not isinstance(key, dict) or not key:
        raise PatchError("key must be a non-empty dict of column=value")

    cols = _table_columns(conn, table)
    if not cols:
        raise PatchError(f"table {table!r} does not exist")
    if field not in cols:
        raise PatchError(f"field {field!r} is not a column on {table}")
    if field in key:
        raise PatchError(
            f"field {field!r} appears in key; patch_field cannot rewrite "
            "an identity column — delete + re-insert with explicit reason instead"
        )
    for col in key:
        if col not in cols:
            raise PatchError(f"key column {col!r} is not on {table}")

    n = _count_matching(conn, table, key)
    if n == 0:
        raise PatchError(f"no row in {table} matches key {key}")
    if n > 1:
        raise PatchError(f"key {key} matches {n} rows in {table}; must be unique")

    old_value = _fetch_current(conn, table, key, field)
    log_path = audit_log_path or _audit_log_path_for(conn)

    if old_value == value:
        return PatchResult(
            table=table,
            key=dict(key),
            field=field,
            old_value=old_value,
            new_value=value,
            reason=reason,
            applied=False,
            audit_log_path=log_path,
            unchanged=True,
        )

    if not apply:
        return PatchResult(
            table=table,
            key=dict(key),
            field=field,
            old_value=old_value,
            new_value=value,
            reason=reason,
            applied=False,
            audit_log_path=log_path,
        )

    where = " AND ".join(f'"{c}" = ?' for c in key)
    update_sql = f'UPDATE "{table}" SET "{field}" = ? WHERE {where}'
    conn.execute(update_sql, (value, *key.values()))

    entry = {
        "ts": _utc_iso_z(),
        "table": table,
        "field": field,
        "key": dict(key),
        "old_value": old_value,
        "new_value": value,
        "reason": reason.strip(),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")

    return PatchResult(
        table=table,
        key=dict(key),
        field=field,
        old_value=old_value,
        new_value=value,
        reason=reason,
        applied=True,
        audit_log_path=log_path,
    )


@dataclass(frozen=True)
class InsertResult:
    """The outcome of an insert_canonical call."""

    table: str
    row: dict[str, Any]
    reason: str
    inserted: bool
    audit_log_path: Path


def insert_canonical(
    table: str,
    row: dict[str, Any],
    reason: str,
    *,
    apply: bool = False,
    conn: sqlite3.Connection,
    audit_log_path: Path | None = None,
) -> InsertResult:
    """Insert a new canonical row with audit logging.

    The companion to :func:`patch_field` for new rows. Per doc §9 the
    reconciler emits ``op='insert'`` for previously-unseen identity
    tuples; this is that primitive.

    Contract:

    - ``table`` must be in :data:`CANONICAL_TABLES`.
    - ``row`` must be a non-empty dict of column→value. Every column in
      ``row`` must exist on the table. The primary key columns must be
      populated.
    - ``reason`` non-empty after whitespace strip.
    - Default dry-run. ``apply=True`` runs the INSERT and appends a
      JSONL audit line with ``op='insert'`` and the full row payload.
    - Raises ``PatchError`` if a row with the same primary key already
      exists. Idempotency on re-emit lives at the reconciler layer
      (line-hash dedup), not here.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise PatchError("reason must be a non-empty string")

    if table not in CANONICAL_TABLES:
        raise PatchError(
            f"table {table!r} is not canonical; "
            f"allowed: {sorted(CANONICAL_TABLES)}"
        )

    if not isinstance(row, dict) or not row:
        raise PatchError("row must be a non-empty dict of column=value")

    cols = _table_columns(conn, table)
    if not cols:
        raise PatchError(f"table {table!r} does not exist")
    for col in row:
        if col not in cols:
            raise PatchError(f"column {col!r} is not on {table}")

    required_pk = _required_pk_columns(conn, table)
    missing_pk = [c for c in required_pk if c not in row]
    if missing_pk:
        raise PatchError(
            f"row is missing primary key columns: {missing_pk}"
        )

    if required_pk:
        pk_key = {c: row[c] for c in required_pk}
        if _count_matching(conn, table, pk_key) > 0:
            raise PatchError(
                f"row with primary key {pk_key} already exists in {table}; "
                "use patch_field for per-field updates"
            )

    log_path = audit_log_path or _audit_log_path_for(conn)

    if not apply:
        return InsertResult(
            table=table,
            row=dict(row),
            reason=reason,
            inserted=False,
            audit_log_path=log_path,
        )

    col_list = ", ".join(f'"{c}"' for c in row)
    placeholders = ", ".join("?" * len(row))
    sql = f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})'
    conn.execute(sql, tuple(row.values()))

    entry = {
        "ts": _utc_iso_z(),
        "op": "insert",
        "table": table,
        "row": dict(row),
        "reason": reason.strip(),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")

    return InsertResult(
        table=table,
        row=dict(row),
        reason=reason,
        inserted=True,
        audit_log_path=log_path,
    )


def _required_pk_columns(
    conn: sqlite3.Connection, table: str
) -> list[str]:
    """Return PK columns the caller must supply (excludes auto-assigned).

    SQLite's ``INTEGER PRIMARY KEY`` (single column) is a ROWID alias and
    auto-assigns when omitted. Other PK shapes (composite, TEXT, etc.)
    must be provided.
    """
    rows = list(conn.execute(f"PRAGMA table_info({table})"))
    # (cid, name, type, notnull, dflt_value, pk)
    pk_rows = [
        (int(r[5]), r[1], str(r[2]).upper()) for r in rows if int(r[5]) > 0
    ]
    pk_rows.sort()
    if len(pk_rows) == 1 and pk_rows[0][2].startswith("INTEGER"):
        # Single INTEGER PRIMARY KEY = ROWID alias, auto-assigned
        return []
    return [name for _, name, _ in pk_rows]


def ensure_anchor(
    conn: sqlite3.Connection,
    label: str,
    bank: str,
    *,
    source: str,
    reason: str,
    category: str = "",
    apply: bool = True,
    audit_log_path: Path | None = None,
) -> int:
    """Get-or-insert an anchor with audit logging on creation.

    Anchor vocabulary is canonical (doc §13-§15) — new entries land in
    the audit log via :func:`insert_canonical`. Existing entries are
    returned without modification (no audit entry, no DB write).

    Bank mismatch on an existing anchor logs a warning and keeps the
    original assignment to avoid semantic drift (same behavior as the
    legacy ``db.get_or_create_anchor`` for compatibility).

    Returns the anchor_id.
    """
    if not isinstance(label, str) or not label:
        raise PatchError("label must be a non-empty string")
    if not isinstance(source, str) or not source.strip():
        raise PatchError("source must be a non-empty string")

    row = conn.execute(
        "SELECT anchor_id, bank FROM anchors WHERE label = ?", (label,)
    ).fetchone()
    if row is not None:
        existing_bank = row[1] if not isinstance(row, sqlite3.Row) else row["bank"]
        existing_id = row[0] if not isinstance(row, sqlite3.Row) else row["anchor_id"]
        if existing_bank != bank:
            import logging

            logging.getLogger(__name__).warning(
                "Anchor %r already assigned to bank %r; ignoring reassignment to %r",
                label,
                existing_bank,
                bank,
            )
        return int(existing_id)

    result = insert_canonical(
        "anchors",
        {
            "label": label,
            "bank": bank,
            "category": category,
            "source": source,
        },
        reason=reason,
        apply=apply,
        conn=conn,
        audit_log_path=audit_log_path,
    )
    if not result.inserted:
        # Dry-run: the row wasn't written; we cannot return a real anchor_id.
        # Caller in dry-run mode should not be calling ensure_anchor in a
        # context that needs the id; raise to surface the misuse.
        raise PatchError(
            "ensure_anchor called with apply=False on a new anchor; "
            "no row inserted, cannot return anchor_id"
        )
    # Fetch the newly-inserted anchor_id
    new_row = conn.execute(
        "SELECT anchor_id FROM anchors WHERE label = ?", (label,)
    ).fetchone()
    if new_row is None:  # pragma: no cover — insert_canonical succeeded
        raise PatchError("insert_canonical reported success but anchor not found")
    return int(new_row[0] if not isinstance(new_row, sqlite3.Row) else new_row["anchor_id"])


def rotate_audit_log(
    path: Path | None = None,
    *,
    max_bytes: int = 100_000_000,
    archive_dir: Path | None = None,
) -> Path | None:
    """Rotate the active audit log if it exceeds ``max_bytes``.

    When triggered: append the active log's contents to
    ``<archive_dir>/patches.archive.jsonl.gz`` (creating or merging-into
    the gzipped archive), then truncate the active log. Returns the
    archive path on rotation, or None if the log was missing/empty/below
    threshold.

    Idempotent: calling on an empty or missing file is a no-op. Per doc
    §67, the archive is also git-trackable (~5 MB gzipped per 100 MB raw)
    for durability.
    """
    p = path or DEFAULT_AUDIT_LOG_PATH
    if not p.exists():
        return None
    size = p.stat().st_size
    if size == 0:
        return None
    if size < max_bytes:
        return None

    archive_parent = archive_dir or (p.parent / "patches-archive")
    archive_parent.mkdir(parents=True, exist_ok=True)
    archive_path = archive_parent / "patches.archive.jsonl.gz"

    # Append-merge: read existing archive, concatenate, rewrite. For very
    # large archives this could be optimized to stream-append, but gzip
    # member concatenation is well-defined and the rotation is rare
    # enough that the simple read-write is fine.
    existing: bytes = b""
    if archive_path.exists():
        with gzip.open(archive_path, "rb") as gz_in:
            existing = gz_in.read()

    new_content = p.read_bytes()
    if not new_content.endswith(b"\n"):
        new_content += b"\n"

    with gzip.open(archive_path, "wb") as gz_out:
        if existing:
            if not existing.endswith(b"\n"):
                existing += b"\n"
            gz_out.write(existing)
        gz_out.write(new_content)

    # Truncate active log
    p.write_bytes(b"")
    return archive_path


def read_audit_log(path: Path | None = None) -> list[dict[str, Any]]:
    """Read and parse the audit log as a list of dicts.

    Used by tests and the (future) reconciler/rollback tooling. Returns an
    empty list when the file does not exist yet.
    """
    p = path or DEFAULT_AUDIT_LOG_PATH
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def read_audit_archive(path: Path) -> list[dict[str, Any]]:
    """Read and parse a rotated gzipped audit archive."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
