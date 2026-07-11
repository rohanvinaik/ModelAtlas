#!/usr/bin/env bash
# Periodic Phase E merge: post-process worker output, apply merge idempotently.
#
# Designed to run under launchd every N hours while the 4 local Phase E workers
# stream to results_{0,1,2,3}.jsonl. Safe to run mid-stream — the workers
# open results in append mode; this script reads a snapshot in time and merges
# whatever's there. All operations are idempotent by (model_id, anchor_id):
#   - merge_phase_e skips anchors already at higher-or-equal confidence
#   - post-processor overwrites its _cleaned outputs each pass
#
# Logs stats + returns 0 on success (launchd re-fires next interval).
# On failure, logs to stderr; launchd re-fires anyway.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
WORK_DIR="${WORK_DIR:-${HOME}/.cache/model-atlas/phase_e_work}"
LOG_FILE="${LOG_FILE:-/tmp/modelatlas-phase_e-merge_periodic.log}"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

# Rotate log at 2 MB to keep it bounded.
if [[ -f "$LOG_FILE" ]] && [[ $(stat -f%z "$LOG_FILE" 2>/dev/null || echo 0) -gt 2097152 ]]; then
    mv "$LOG_FILE" "${LOG_FILE}.1"
fi

exec >>"$LOG_FILE" 2>&1

log "=== merge_phase_e_periodic start ==="

# 1. Discover worker output files (results_*.jsonl at top level).
# Excludes _cleaned outputs (post-processor writes those) and archived subdirs.
shopt -s nullglob
INPUTS=()
for f in "${WORK_DIR}"/results_*.jsonl; do
    base="$(basename "$f")"
    if [[ "$base" == *_cleaned*.jsonl ]]; then
        continue
    fi
    if [[ -s "$f" ]]; then
        n=$(wc -l < "$f" | tr -d ' ')
        log "  $base: ${n} lines"
        INPUTS+=("$f")
    fi
done
shopt -u nullglob

if [[ ${#INPUTS[@]} -eq 0 ]]; then
    log "no input files to merge; exiting"
    exit 0
fi

CLEANED=()
for f in "${INPUTS[@]}"; do
    log "postprocess: $(basename "$f")"
    "$PYTHON" "${REPO_ROOT}/scripts/phase_e_postprocess.py" --input "$f" 2>&1 | tail -6
    # postprocess writes ${f%.jsonl}_cleaned.jsonl
    cleaned="${f%.jsonl}_cleaned.jsonl"
    if [[ -s "$cleaned" ]]; then
        CLEANED+=("$cleaned")
    else
        log "  WARN: cleaned output missing for $f"
    fi
done

if [[ ${#CLEANED[@]} -eq 0 ]]; then
    log "no cleaned files produced; exiting"
    exit 0
fi

# 3. Dry-run merge (visibility on what would change).
log "merge dry-run:"
"$PYTHON" -m model_atlas.ingest_cli --merge-e "${CLEANED[@]}" --merge-e-dry-run 2>&1 | tail -3

# 4. Apply merge (idempotent).
log "merge apply:"
"$PYTHON" -m model_atlas.ingest_cli --merge-e "${CLEANED[@]}" 2>&1 | tail -3

log "=== merge_phase_e_periodic end ==="
