# Reconciler — JSONL → canonical DB with line-hash idempotency

The reconciler is the bridge between distributed worker output and
canonical reference tables. Workers emit JSONL to a known location; the
reconciler walks those files and applies each row through the
audit-logged write primitives (`insert_canonical` for new rows,
`patch_field` for per-field updates on existing ones).

See `PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md` §9-§10 for the doctrine.

## Why a reconciler

Without one:

- Workers writing directly to canonical tables race the interactive
  admin path and the audit log.
- There's no chokepoint where bad rows can be caught before they land.
- Re-running a merge after a worker bug requires manual deduplication.

With one:

- Workers are stateless beyond their own crawl-state. Adding a worker is
  "rsync the code, install the cron"; removing one is "stop the cron."
- Re-running the reconciler against the same JSONL files is a no-op
  (line-hash idempotency).
- Every canonical change carries a sourced reason in the audit log.

## JSONL line shape

```json
{
  "op": "upsert",
  "table": "models",
  "key": {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
  "row": {"author": "meta-llama", "source": "huggingface", "display_name": "Llama 3.1 8B Instruct"},
  "source_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
  "host": "macpro",
  "captured_at": "2026-05-15T22:31:04Z"
}
```

Fields:

- `op` — one of `"upsert"` (default), `"insert"`, `"patch"`. `upsert`
  inserts if the key is absent and patches if present. `insert` errors
  if a row with the key exists. `patch` errors if no row matches.
- `table` — must be in `CANONICAL_TABLES` (`models`, `model_positions`,
  `model_links`, `anchors`).
- `key` — identity tuple. Must uniquely identify the row.
- `row` — fields to apply. Key columns may appear here and are merged
  into `key` for insert.
- `source_url`, `host`, `captured_at` — provenance fields. Combined into
  the audit log `reason` so every canonical change is traceable to the
  worker that produced it.

## Idempotency via line-hash

The reconciler computes `sha256(raw_line_bytes)[:32]` for each line. If
the hash is already in the `reconciler_processed` table, the line is
skipped. This means:

- **Replay is safe.** Re-running on the same file produces zero new
  patches.
- **Partial runs are safe.** If the reconciler crashes mid-file, the
  next run picks up at the first unprocessed line.
- **Worker resends are safe.** Crash-restart or back-fill that re-emits
  the same line is a no-op.

The hash is line-level, not content-level. A line that differs only by
`captured_at` is a different hash and runs through, but the field-diff
step will detect no changes and record an "unchanged" outcome instead
of writing an audit entry.

## Dry-run

`apply=False` plans the pass without writing anything — no DB updates,
no audit lines, no processed-marking. Use this to preview a worker
batch before committing.

```python
from pathlib import Path
import sqlite3
from model_atlas import db
from model_atlas.reconciler import reconcile_file

conn = db.get_connection()
stats = reconcile_file(Path("data/incoming/macpro/phase_b_2026-05-16.jsonl"), conn)
print(stats)
# ReconcileStats(file='...', lines_seen=523, inserts=12, patches=89, unchanged=422, errors=[])

# Looks good — commit it
stats = reconcile_file(
    Path("data/incoming/macpro/phase_b_2026-05-16.jsonl"),
    conn,
    apply=True,
)
conn.commit()
```

## Per-host directory layout

Per doc §12, worker output should be isolated by host:

```
data/incoming/
├── macpro/
│   ├── phase_b_2026-05-15.jsonl
│   └── phase_b_2026-05-16.jsonl
└── homebridge/
    └── phase_c_2026-05-16.jsonl
```

The reconciler walks one file at a time. A sync-and-reconcile script
should loop over hosts:

```bash
for host in macpro homebridge; do
    rsync "$host:~/model-atlas/data/incoming/$host/" "data/incoming/$host/"
    for f in data/incoming/$host/*.jsonl; do
        python -c "from pathlib import Path; from model_atlas import db; from model_atlas.reconciler import reconcile_file; \
            conn = db.get_connection(); print(reconcile_file(Path('$f'), conn, apply=True)); conn.commit()"
    done
done
```

This wrapper script does not yet exist in ModelAtlas; the reconciler
primitive does. Operationalizing the morning sync routine is a
follow-on.

## What the reconciler does NOT do

- **Worker code changes**: existing Phase C and Phase D worker scripts
  emit JSONL in their own shapes (see `scripts/phase_e_worker.py` and
  the C-tier workers). Routing those through the reconciler requires
  either (a) emitting in the reconciler's expected shape, or (b) a
  shape-translator stage.
- **Legacy merge replacement**: `ingest_phase_c_merge.py` and
  `phase_d_merge.py` still operate. The reconciler is a *new* path
  for new workers; migrating existing merges is a separate refactor.
- **Backfill of audit log for past merges**: rows already in the DB
  have no audit-log entries. The reconciler does not back-fill the
  history.

## See also

- `PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md` §9-§12, §45-§50.
- `src/model_atlas/reconciler.py` — the implementation.
- `tests/test_reconciler.py` — contract pinned by tests.
- [`admin.md`](admin.md) — the `patch_field` + `insert_canonical`
  primitives the reconciler dispatches through.
