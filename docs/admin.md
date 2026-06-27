# Admin primitives — audit-logged writes

ModelAtlas follows the persistent-knowledge architecture described in
`PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md`. This document covers the
write primitive that enforces audit-log discipline on canonical reference
data.

## Bi-modal split: canonical vs. observation

| Mode | Tables | Write path | Cadence |
|------|--------|------------|---------|
| **Canonical reference** (Mode 1) | `models`, `model_positions`, `model_links`, `anchors` | `patch_field()` — sourced, dry-run-default | Slow; human-curated or reconciler-driven |
| **Observation / derived** (Mode 2) | `model_metadata`, `model_anchors`, `audit_findings`, `correction_events`, `phase_d_runs` | Direct insert/update from extractors and Phase D pipeline | Fast; worker-driven |

The split is the load-bearing invariant. Canonical rows describe *what
exists* (a model's identity, its bank positions, the anchor vocabulary).
Observation rows describe *what we saw* (extracted vibes, Phase D mismatch
findings, derived confidence scores). Conflating them poisons reference
data — every direct write to a canonical table must justify a sourced
reason.

`CANONICAL_TABLES` in `src/model_atlas/admin.py` is the source of truth.
Adding a table is an architectural decision; coordinate with the doc.

## `patch_field` — the single canonical write

```python
from model_atlas.admin import patch_field

result = patch_field(
    "models",
    {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
    "author",
    "Meta",
    reason="Confirmed via HF API 2026-05-16; org name canonicalized",
    apply=True,
    conn=conn,
)
```

Contract:

- `table` must be in `CANONICAL_TABLES`.
- `key` must uniquely identify exactly one existing row (raises `PatchError`
  if zero or many match).
- `field` must be a real column on `table` and **not** in the `key`.
- `reason` must be non-empty after whitespace strip — `"update"`, `"fix"`,
  `"correct"` are anti-patterns the audit log explicitly exists to prevent.
- Default is **dry-run**: the call returns a `PatchResult` showing
  `old_value` → `new_value` without modifying anything. The cost of
  accidentally calling `patch_field` is zero until `apply=True` is set.
- `apply=True` runs the `UPDATE` and appends a JSONL line to the audit log.
  Both happen inside the caller's transaction — `patch_field` never
  commits on the caller's behalf.
- When `old == new`, the call is a no-op (`unchanged=True`, no audit line
  written, no DB write).

### Why single-row, single-field?

Bulk updates without per-row justification are the failure mode this
primitive eliminates. If you need to change ten fields on one row, write
ten `patch_field` calls with ten reasons. If you need to change one field
on ten rows, write ten calls. The friction is the point.

## Audit log location

The audit log is **append-only JSONL**. It lives **outside the SQLite
database** so a corrupted DB doesn't lose the provenance trail.

- **Canonical DB** (`~/.cache/model-atlas/network.db`) → audit log at
  `<repo>/data/patches.jsonl`. Git-tracked.
- **Test or staging DB** (file path elsewhere) → audit log co-located:
  `<db_dir>/patches.jsonl`.
- **`:memory:` connections** require the `MODEL_ATLAS_PATCHES_PATH`
  environment variable or an explicit `audit_log_path=` argument.

Line shape:

```json
{
  "ts": "2026-05-16T14:23:11Z",
  "table": "models",
  "field": "author",
  "key": {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
  "old_value": "meta-llama",
  "new_value": "Meta",
  "reason": "Confirmed via HF API 2026-05-16; org name canonicalized"
}
```

The `reason` field is the most important column. A patch without a reason
is a regression waiting to happen — six months later the reader needs to
decide whether the patch is still trustworthy.

Good reasons:

- ✅ `"Confirmed via HF API 2026-05-16; org name canonicalized"`
- ✅ `"Direct observation, ran inference and verified 7B parameter count"`
- ✅ `"Reconciler pass 2026-05-10 from data/incoming/macpro/phase_b_2026-05-10.jsonl"`

Bad reasons:

- ❌ `"update"` / `"fix"` / `"correct"` / `"improve"`

## `insert_canonical` — audit-logged insertions

Companion to `patch_field` for *new* canonical rows. Same dry-run-default
contract; the audit log entry carries `"op": "insert"` and the full row
payload.

```python
from model_atlas.admin import insert_canonical

result = insert_canonical(
    "models",
    {"model_id": "Qwen/Qwen2.5-Coder-1.5B", "author": "Qwen",
     "source": "huggingface", "display_name": "Qwen2.5-Coder-1.5B"},
    reason="Phase A seed from HF API 2026-05-16",
    apply=True,
    conn=conn,
)
```

Contract additions on top of the shared invariants:

- Primary-key columns must be present in `row`.
- A row with that primary key must not already exist — duplicate PK
  raises `PatchError`. Idempotency on re-emit lives at the reconciler
  layer (line-hash dedup), not here.
- INSERT and audit-log append happen inside the caller's transaction.

## Audit log rotation

`rotate_audit_log(path, max_bytes=100_000_000)` rotates the active log
into `data/patches-archive/patches.archive.jsonl.gz` when it exceeds the
threshold. The archive is append-merged on subsequent rotations; the
active log is truncated to empty. Per doc §67, the gzipped archive is
~5 MB per 100 MB raw — cheap to keep git-tracked for durability.

Rotation is a no-op when the file is missing, empty, or below threshold.

## Reconciler

`model_atlas.reconciler.reconcile_file(path, conn, apply=False)` walks
worker-emitted JSONL and applies each line via `insert_canonical` (new
rows) or `patch_field` (existing rows). See [`reconciler.md`](reconciler.md)
for the line shape and idempotency guarantees.

The reconciler is the *integration* of `patch_field` + `insert_canonical`
with line-hash dedup — it's how workers should write to canonical data
without bypassing the audit log.

## Coherence audit

`python -m model_atlas.coherence` produces a health report covering
bank orthogonality, NULL coverage, anchor orphans/oversaturation, and
uncited audit-log entries. Read-only — never modifies the DB. See
[`coherence.md`](coherence.md).

## What this primitive does NOT do (yet)

Per the persistent-knowledge doc, the full discipline has additional
pieces that are not yet implemented in ModelAtlas:

- **Routing existing phase merges through the reconciler**:
  `ingest_phase_c_merge.py`, `phase_d_merge.py`, and
  `scripts/phase_e_postprocess.py` still write directly. The primitives
  now exist; migrating each call site is a per-site judgment call.
- **Sanctioned-exception list with named exceptions**: see
  `.claude/CLAUDE.md`. None registered yet — legacy paths are
  pre-existing, not sanctioned.
- **`bank-derive --explain <model_id>`**: per-model derivation walk
  showing each rule's inputs and output. Requires design discussion
  about how to surface the current inline derivation logic.

## See also

- `PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md` §7-§12 — the doctrine
  this primitive implements.
- [`reconciler.md`](reconciler.md), [`coherence.md`](coherence.md).
- `src/model_atlas/admin.py` — the implementation.
- `tests/test_admin.py` — the contract pinned by tests.
