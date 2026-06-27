"""JSONL reconciler with line-hash idempotency.

Walks worker-emitted JSONL files and applies each row to canonical
reference tables via :func:`model_atlas.admin.insert_canonical` (for new
rows) or :func:`model_atlas.admin.patch_field` (for per-field diffs on
existing rows). Re-running the reconciler on the same files is safe —
processed line-hashes are tracked in ``reconciler_processed``.

This is the *primitive*. It does not yet replace existing phase merges
(``ingest_phase_c_merge.py``, ``phase_d_merge.py``) — migrating those is
a separate effort once the primitive has bedded in.

JSONL line shape (per ``PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md`` §9):

.. code-block:: json

   {
     "op": "upsert",
     "table": "models",
     "key": {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
     "row": {"author": "meta-llama", "source": "huggingface", ...},
     "source_url": "https://huggingface.co/...",
     "host": "macpro",
     "captured_at": "2026-05-15T22:31:04Z"
   }

The line-hash is SHA-256 of the *raw line bytes* (including trailing
newline as it was read), truncated to 32 hex chars. Different
``captured_at`` for otherwise identical content produces different
hashes and both pass through — the reconciler is responsible for
noticing the second produces no diffs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .admin import (
    CANONICAL_TABLES,
    PatchError,
    insert_canonical,
    patch_field,
)

logger = logging.getLogger(__name__)


_RECONCILER_PROCESSED_SCHEMA = """
CREATE TABLE IF NOT EXISTS reconciler_processed (
    line_hash    TEXT PRIMARY KEY,
    source_file  TEXT,
    processed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_reconciler_source
    ON reconciler_processed(source_file);
"""


@dataclass
class ReconcileStats:
    """Summary of one reconciler pass over a file."""

    file: str
    lines_seen: int = 0
    lines_skipped_processed: int = 0
    lines_skipped_malformed: int = 0
    inserts: int = 0
    patches: int = 0
    unchanged: int = 0
    errors: list[str] = field(default_factory=list)


def ensure_reconciler_schema(conn: sqlite3.Connection) -> None:
    """Idempotently create the ``reconciler_processed`` table."""
    conn.executescript(_RECONCILER_PROCESSED_SCHEMA)


def _line_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:32]


def _already_processed(conn: sqlite3.Connection, line_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM reconciler_processed WHERE line_hash = ?", (line_hash,)
    ).fetchone()
    return row is not None


def _mark_processed(
    conn: sqlite3.Connection, line_hash: str, source_file: str
) -> None:
    from datetime import datetime, timezone

    conn.execute(
        "INSERT OR IGNORE INTO reconciler_processed (line_hash, source_file, processed_at) "
        "VALUES (?, ?, ?)",
        (
            line_hash,
            source_file,
            datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
                "+00:00", "Z"
            ),
        ),
    )


def _build_reason(item: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.append("reconciler")
    if host := item.get("host"):
        parts.append(f"host={host}")
    if src := item.get("source_url"):
        parts.append(f"src={src}")
    if ts := item.get("captured_at"):
        parts.append(f"captured_at={ts}")
    return "; ".join(parts)


def _validate_item(item: Any) -> str | None:
    """Return None if valid, else an error string."""
    if not isinstance(item, dict):
        return "not a dict"
    table = item.get("table")
    if not isinstance(table, str) or not table:
        return "missing or invalid 'table'"
    if table not in CANONICAL_TABLES:
        return f"table {table!r} is not canonical"
    op = item.get("op", "upsert")
    if op not in {"upsert", "insert", "patch"}:
        return f"unknown op {op!r}"
    if not isinstance(item.get("key"), dict) or not item["key"]:
        return "missing or invalid 'key'"
    if not isinstance(item.get("row"), dict):
        return "missing or invalid 'row'"
    return None


def _iter_raw_lines(path: Path) -> Iterator[tuple[bytes, int]]:
    """Yield (raw_bytes, 1-indexed line number) for each non-empty line."""
    with path.open("rb") as f:
        for n, raw in enumerate(f, start=1):
            if raw.strip():
                yield raw, n


def reconcile_items(
    items: list[dict[str, Any]],
    conn: sqlite3.Connection,
    *,
    apply: bool = False,
    audit_log_path: Path | None = None,
    source_label: str = "in_process",
) -> ReconcileStats:
    """Apply an in-memory list of items to canonical tables.

    Same per-item logic as :func:`reconcile_file` but skips the
    line-hash idempotency (no file). Used by in-process flows like
    :func:`model_atlas.extraction.pipeline.extract_and_store` that need
    to dispatch reconciler-shaped items without a JSONL round-trip.

    ``source_label`` is recorded as the file in :class:`ReconcileStats`
    and used as the ``host`` fallback when items omit it.
    """
    stats = ReconcileStats(file=source_label)
    for idx, item in enumerate(items, start=1):
        stats.lines_seen += 1
        err = _validate_item(item)
        if err:
            stats.lines_skipped_malformed += 1
            stats.errors.append(f"item {idx}: {err}")
            continue

        table = item["table"]
        key: dict[str, Any] = item["key"]
        row: dict[str, Any] = item["row"]
        op = item.get("op", "upsert")
        reason = _build_reason(item) if item.get("host") or item.get(
            "source_url"
        ) or item.get("captured_at") else f"reconciler ({source_label})"

        try:
            _reconcile_one(
                conn,
                table=table,
                key=key,
                row=row,
                op=op,
                reason=reason,
                apply=apply,
                stats=stats,
                audit_log_path=audit_log_path,
            )
        except PatchError as e:
            stats.errors.append(f"item {idx}: {e}")
            continue
    return stats


def reconcile_file(
    path: Path,
    conn: sqlite3.Connection,
    *,
    apply: bool = False,
    audit_log_path: Path | None = None,
) -> ReconcileStats:
    """Apply a worker JSONL file to canonical tables.

    For each line:

    1. Compute the line-hash. Skip if already in ``reconciler_processed``.
    2. Parse JSON. Validate shape. Skip malformed.
    3. Look up the row by ``key``:

       - If absent and op in {"upsert", "insert"}: ``insert_canonical``
         with the merged ``key | row`` payload.
       - If present: for each field in ``row``, ``patch_field`` if the
         value differs.
    4. Mark the line-hash processed (when ``apply=True``).

    Default ``apply=False`` is dry-run for the entire pass — no writes,
    no audit log, no processed-marking. Use this to preview before
    committing. Caller commits/rolls back the transaction.
    """
    ensure_reconciler_schema(conn)

    stats = ReconcileStats(file=str(path))
    source_file = path.name

    for raw, line_no in _iter_raw_lines(path):
        stats.lines_seen += 1
        h = _line_hash(raw)
        if _already_processed(conn, h):
            stats.lines_skipped_processed += 1
            continue
        try:
            item = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            stats.lines_skipped_malformed += 1
            stats.errors.append(f"line {line_no}: JSON decode: {e}")
            continue

        err = _validate_item(item)
        if err:
            stats.lines_skipped_malformed += 1
            stats.errors.append(f"line {line_no}: {err}")
            continue

        table = item["table"]
        key: dict[str, Any] = item["key"]
        row: dict[str, Any] = item["row"]
        op = item.get("op", "upsert")
        reason = _build_reason(item)

        try:
            _reconcile_one(
                conn,
                table=table,
                key=key,
                row=row,
                op=op,
                reason=reason,
                apply=apply,
                stats=stats,
                audit_log_path=audit_log_path,
            )
        except PatchError as e:
            stats.errors.append(f"line {line_no}: {e}")
            continue

        if apply:
            _mark_processed(conn, h, source_file)

    return stats


def _reconcile_one(
    conn: sqlite3.Connection,
    *,
    table: str,
    key: dict[str, Any],
    row: dict[str, Any],
    op: str,
    reason: str,
    apply: bool,
    stats: ReconcileStats,
    audit_log_path: Path | None,
) -> None:
    """Apply one validated item. Mutates ``stats`` in place."""
    where = " AND ".join(f'"{c}" = ?' for c in key)
    exists = conn.execute(
        f'SELECT 1 FROM "{table}" WHERE {where}', tuple(key.values())
    ).fetchone()

    if not exists:
        if op == "patch":
            raise PatchError(
                f"op='patch' but no row in {table} matches key {key}"
            )
        # Insert: combine key + row (row may already include key cols)
        payload: dict[str, Any] = dict(row)
        for c, v in key.items():
            payload.setdefault(c, v)
        insert_canonical(
            table,
            payload,
            reason=reason,
            apply=apply,
            conn=conn,
            audit_log_path=audit_log_path,
        )
        stats.inserts += 1
        return

    # Existing row: diff each field in ``row`` and patch where different
    changed = 0
    for col, new_val in row.items():
        if col in key:
            continue  # key columns are identity, never patched here
        result = patch_field(
            table,
            key=key,
            field=col,
            value=new_val,
            reason=reason,
            apply=apply,
            conn=conn,
            audit_log_path=audit_log_path,
        )
        if result.unchanged:
            continue
        changed += 1
    if changed > 0:
        stats.patches += 1
    else:
        stats.unchanged += 1
