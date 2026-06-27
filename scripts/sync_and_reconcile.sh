#!/usr/bin/env bash
# Sync worker JSONL from spokes, reconcile into canonical DB, audit health.
#
# Doctrine: PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md §49.
#
# Operational pattern: hub-and-spoke. Spoke machines emit JSONL to
# ~/model-atlas/data/incoming/<host>/. This script (run on the hub)
# rsyncs their output, applies via the reconciler, then runs the
# coherence audit and rotates the audit log if needed.
#
# Idempotent — safe to re-run at any time.
#
# Usage:
#   ./scripts/sync_and_reconcile.sh                # full pass, applies writes
#   ./scripts/sync_and_reconcile.sh --dry-run      # plan only, no writes
#   ./scripts/sync_and_reconcile.sh --skip-sync    # skip rsync (reconcile local only)

set -euo pipefail

# --- Configuration ------------------------------------------------------------

# Spoke hosts. Hostnames must match Headscale / SSH config aliases. The
# expected per-host JSONL output dir is data/incoming/<host>/ on the hub
# and ~/model-atlas/data/incoming/<host>/ on each spoke.
HOSTS=("macpro" "homebridge")

# Remote and local paths
REMOTE_INCOMING_BASE="~/model-atlas/data/incoming"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_INCOMING_BASE="${REPO_ROOT}/data/incoming"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

DRY_RUN=0
SKIP_SYNC=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --skip-sync) SKIP_SYNC=1 ;;
        -h|--help)
            sed -n '1,/^set/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg" >&2
            exit 2
            ;;
    esac
done

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter not found at $PYTHON" >&2
    echo "Set PYTHON=/path/to/python or create .venv/ at the repo root." >&2
    exit 1
fi

# --- 1. Sync per-host JSONL from spokes ---------------------------------------

if [[ $SKIP_SYNC -eq 0 ]]; then
    for host in "${HOSTS[@]}"; do
        log "rsync: ${host}:${REMOTE_INCOMING_BASE}/${host}/  ->  ${LOCAL_INCOMING_BASE}/${host}/"
        mkdir -p "${LOCAL_INCOMING_BASE}/${host}"
        if [[ $DRY_RUN -eq 1 ]]; then
            rsync -av --update --dry-run \
                "${host}:${REMOTE_INCOMING_BASE}/${host}/" \
                "${LOCAL_INCOMING_BASE}/${host}/" || log "WARN: rsync from ${host} failed"
        else
            rsync -av --update \
                "${host}:${REMOTE_INCOMING_BASE}/${host}/" \
                "${LOCAL_INCOMING_BASE}/${host}/" || log "WARN: rsync from ${host} failed"
        fi
    done
else
    log "sync: skipped (--skip-sync)"
fi

# --- 2. Reconcile per-host (idempotent via line-hash) -------------------------

for host in "${HOSTS[@]}"; do
    host_dir="${LOCAL_INCOMING_BASE}/${host}"
    if [[ ! -d "$host_dir" ]]; then
        log "reconcile: no directory for ${host}, skipping"
        continue
    fi
    shopt -s nullglob
    files=("${host_dir}"/*.jsonl)
    shopt -u nullglob
    if [[ ${#files[@]} -eq 0 ]]; then
        log "reconcile: no JSONL for ${host}"
        continue
    fi
    for f in "${files[@]}"; do
        log "reconcile (${host}): $(basename "$f")"
        if [[ $DRY_RUN -eq 1 ]]; then
            "$PYTHON" -c "
import sys
from pathlib import Path
from model_atlas import db
from model_atlas.reconciler import reconcile_file
conn = db.get_connection()
stats = reconcile_file(Path('$f'), conn, apply=False)
print(f'  DRY: lines_seen={stats.lines_seen} would_insert={stats.inserts} would_patch={stats.patches} unchanged={stats.unchanged} skipped={stats.lines_skipped_processed} errors={len(stats.errors)}')
for e in stats.errors[:5]:
    print(f'    error: {e}', file=sys.stderr)
"
        else
            "$PYTHON" -c "
import sys
from pathlib import Path
from model_atlas import db
from model_atlas.reconciler import reconcile_file
conn = db.get_connection()
stats = reconcile_file(Path('$f'), conn, apply=True)
conn.commit()
print(f'  inserts={stats.inserts} patches={stats.patches} unchanged={stats.unchanged} skipped={stats.lines_skipped_processed} errors={len(stats.errors)}')
for e in stats.errors[:5]:
    print(f'    error: {e}', file=sys.stderr)
"
        fi
    done
done

# --- 2b. Legacy C-phase result pull + merge ----------------------------------
# Workers on macpro/homebridge write to legacy paths (NOT data/incoming/<host>/).
# This block scp's their result JSONLs back and merges via ingest_phase_c_merge.
# Direct Mode-2 writes (model_metadata) — not audit-logged. Idempotent: merge
# uses INSERT OR REPLACE / skip-existing semantics.

declare -A LEGACY_PULLS=(
    [macpro:/Users/squishfam/phase_c_work/results_0.jsonl]="${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_c_work/results_0.jsonl"
    [macpro:/Users/squishfam/phase_c_work/results_0_remaining.jsonl]="${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_c_work/results_0_remaining.jsonl"
    [macpro:/Users/squishfam/phase_c1/results_c1.jsonl]="${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_c1_work/results_c1_extended.jsonl"
    [homebridge:~/phase_c/results_c3_1.jsonl]="${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_c3_work/results_c3_1.jsonl"
    # Phase E results
    [macpro:/Users/squishfam/phase_e/results_2.jsonl]="${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_e_work/phase_e_results_2.jsonl"
    [macpro:/Users/squishfam/phase_e/results_3.jsonl]="${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_e_work/phase_e_results_3.jsonl"
    [homebridge:~/phase_e/results_4.jsonl]="${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_e_work/phase_e_results_4.jsonl"
    [homebridge:~/phase_e/results_5.jsonl]="${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_e_work/phase_e_results_5.jsonl"
)

if [[ $SKIP_SYNC -eq 0 ]]; then
    log "legacy C-phase pull from spokes"
    for src in "${!LEGACY_PULLS[@]}"; do
        dst="${LEGACY_PULLS[$src]}"
        mkdir -p "$(dirname "$dst")"
        if [[ $DRY_RUN -eq 1 ]]; then
            log "  DRY: would scp $src -> $dst"
        else
            scp -q "$src" "$dst" 2>/dev/null \
                && log "  pulled $src" \
                || log "  WARN: scp $src failed (worker not deployed or file missing)"
        fi
    done
else
    log "legacy C-phase pull: skipped (--skip-sync)"
fi

if [[ $DRY_RUN -eq 0 ]]; then
    log "legacy C-phase merge"
    # Each merge call is idempotent. Missing files are skipped (FileNotFoundError handled).
    for cmd in \
        "--merge-c1 ${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_c1_work/results_c1_extended.jsonl" \
        "--merge-c2 ${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_c_work/results_0.jsonl ${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_c_work/results_0_remaining.jsonl" \
        "--merge-c3 ${CACHE_DIR:-$HOME/.cache/model-atlas}/phase_c3_work/results_c3_1.jsonl"
    do
        # Check at least one file exists before attempting merge
        files=$(echo "$cmd" | tr -s ' ' '\n' | grep '\.jsonl$' || true)
        any_present=0
        for f in $files; do
            [[ -s "$f" ]] && any_present=1
        done
        if [[ $any_present -eq 1 ]]; then
            log "  merging: $cmd"
            "$PYTHON" -m model_atlas.ingest_cli $cmd 2>&1 | tail -3 || log "  WARN: merge failed"
        fi
    done
else
    log "legacy C-phase merge: skipped (--dry-run)"
fi

# --- 3. Run coherence audit ---------------------------------------------------

log "coherence audit"
"$PYTHON" -m model_atlas.coherence

# --- 4. Rotate audit log if needed --------------------------------------------

log "audit log rotation check"
"$PYTHON" -c "
from model_atlas.admin import rotate_audit_log
result = rotate_audit_log()
if result is None:
    print('  no rotation needed')
else:
    print(f'  rotated to: {result}')
"

log "sync_and_reconcile complete"
