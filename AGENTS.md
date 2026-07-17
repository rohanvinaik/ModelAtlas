# AGENTS.md

## Scope
- Applies to the entire `model-atlas` repository.
- Follow user instructions first, then this file, then local nested guidance.

## Execution Contract
- Read relevant files before editing.
- Prefer minimal diffs over broad rewrites.
- Avoid behavior changes unless requested or required to fix defects.
- Surface assumptions and risks when information is incomplete.

## Required Validation
- Run the smallest check set that proves the change is correct.
- `uv run ruff check .`
- `uv run pytest -q`
- `uv run mypy .`

## Theory and Context
- Read `CLAUDE.md` and `.claude/rules/theory.md` before deep refactors.
- Keep implementation aligned with: Why signed hierarchies instead of flat categories: A categorical "size" field with values {small, medium, large} can't express proximity.
- If work conflicts with explicit rules, stop and request clarification.

## Key Components

### Write discipline (canonical tables only — `models`, `model_positions`, `model_links`, `anchors`)
- `src/model_atlas/admin.py` — `patch_field()`, `insert_canonical()`, `ensure_anchor()`. Dry-run by default. Every write appends to `data/patches.jsonl`.
- `src/model_atlas/reconciler.py` — `reconcile_file()` for worker JSONL → primitives, idempotent via SHA-256 line hash.
- `src/model_atlas/coherence.py` — read-only audit: bank orthogonality, NULL coverage, anchor orphans/oversaturation, uncited canonical writes. Run weekly: `python -m model_atlas.coherence`.
- `scripts/sync_and_reconcile.sh` — hub-and-spoke sync wrapper: rsync from spokes → reconciler → coherence → audit-log rotate. Idempotent.

### Tool CLI
- The user-facing CLI is `model_atlas.ingest_cli`, not `model_atlas.ingest`. The latter has no CLI dispatch.
- MCP tool surface: `src/model_atlas/server.py` (10 tools). When adding/removing tools, update `AGENTS.md`, `README.md`, and `docs/DESIGN.md` in the same change. Verify count with `grep -Rho "@mcp.tool()" src/model_atlas/server.py | wc -l`.

### The refinement loop (`navigate_models` → `refine`)
- `navigate_models` returns a `refine` block beside `results`. Built by
  `build_refinement_guidance()` in `src/model_atlas/query_navigate.py`; serialized by
  `_refine_payload()` in `server.py`. Pure function of the returned window + the query —
  same inputs always give the same guidance.
- **The contract**: `refine.question` names the highest-value unspecified dimension.
  Each option carries an `apply` dict the caller MERGES into the arguments it already
  sent — scalars replace, lists append. Callers must not rebuild the query.
- `question_id` is the stable machine contract; `question` prose is free to change.
  Skeletons live in `QUESTION_TEMPLATES` — declarative, `<slot>`-gapped, same idiom as
  the certifier's `Rule.reason_template`. `render_question()` raises `KeyError` on an
  unfilled slot, so a literal `<bank>` can never reach a payload.
- **Two invariants, both test-pinned** (`tests/test_navigate_refine.py`):
  1. An option must never point outside the window's observed range — answering it
     would return an empty set. Options come from `_axis_options(bank, lo, hi)`.
  2. A bank whose window is uniform is dropped, not asked about.
- `ranking_degraded: true` ⇔ the query supplied no `prefer_anchors`. Three of five soft
  signals (PMI-match, rare-boost, superadditive) are then constant across every
  candidate that cleared the `require` filter, so the window is FILTERED but not
  ORDERED. This is a real property of the scoring layer, not a warning to paper over.

### Legacy write paths
`ingest_phase_c_merge.py`, `phase_d_heal.py`, `phase_d_merge.py`, `scripts/phase_e_postprocess.py`, `ingest.py` Phase A/B, `ingest_seed.py` — these write canonical tables directly via `db.insert_model`, `db.set_position`, `db.add_link`. Pre-existing, *not* sanctioned. Do not copy their patterns into new code. Migrating each to the reconciler is a per-site judgment call (see `.claude/CLAUDE.md` → Sanctioned write exceptions).

### Doc map
| Topic | File |
|-------|------|
| Architecture deep dive | `docs/DESIGN.md` |
| Pipeline (Phases A–E) | `docs/pipeline.md` |
| Audit-logged primitives | `docs/admin.md` |
| Reconciler | `docs/reconciler.md` |
| Coherence audit | `docs/coherence.md` |
| Hub-and-spoke deployment | `deploy/README.md`, `deploy/phase_e/README.md` |
| Persistent-knowledge doctrine | `PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md` (external) |

## Handoff Expectations
- Summarize what changed and why.
- Report what was tested and what remains unverified.
